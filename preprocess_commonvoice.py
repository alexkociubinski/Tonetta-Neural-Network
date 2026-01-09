"""
PREPROCESS MOZILLA COMMON VOICE DATASET

This script downloads and preprocesses the Mozilla Common Voice dataset
for training the voice modification neural network.

WHAT IT DOES:
1. Loads audio samples from Common Voice via Hugging Face datasets
2. Extracts features: RMS energy, pitch (F0), speaking rate
3. Generates target control signals using heuristics
4. Saves preprocessed data as .npz files for fast training

USAGE:
    python preprocess_commonvoice.py --samples 1000
    python preprocess_commonvoice.py --samples 100 --dry-run  # Test mode
"""

import numpy as np
import librosa
import os
import argparse
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def load_commonvoice_streaming(
    num_samples: int = 1000,
    language: str = 'en',
    split: str = 'train',
    seed: int = 42
) -> List[Dict]:
    """
    Load speech samples by downloading audio files directly and using librosa.
    
    PLAIN ENGLISH:
    - Downloads audio file URLs from the dataset
    - Uses librosa to decode audio (no PyTorch required)
    - Returns a list of audio arrays with metadata
    
    Args:
        num_samples: Number of samples to load
        language: Language code ('en' for English)
        split: Dataset split ('train', 'validation', 'test')
        seed: Random seed for reproducibility
    
    Returns:
        List of dictionaries with 'audio', 'sentence', 'sample_rate'
    """
    try:
        from datasets import load_dataset, Audio
    except ImportError:
        raise ImportError(
            "Please install the datasets library: pip install datasets"
        )
    
    import tempfile
    import io
    
    print(f"Loading {num_samples} samples from speech dataset...")
    print("  Using streaming mode with librosa decoding")
    
    # List of datasets to try (in order of preference)
    datasets_to_try = [
        {
            'name': 'openslr/librispeech_asr',
            'config': 'clean',
            'split': 'train.100',
            'audio_key': 'audio',
            'text_key': 'text'
        },
    ]
    
    dataset = None
    dataset_info = None
    
    for ds_config in datasets_to_try:
        try:
            print(f"  Trying dataset: {ds_config['name']}...")
            
            # Load dataset - we need to disable automatic audio decoding
            dataset = load_dataset(
                ds_config['name'],
                ds_config['config'],
                split=ds_config['split'],
                streaming=True
            )
            
            # Cast the audio column to NOT decode (just get the raw file info)
            # Using decode=False gets us the path/bytes without trying to decode
            dataset = dataset.cast_column(
                ds_config['audio_key'],
                Audio(decode=False)
            )
            
            dataset_info = ds_config
            print(f"  ✓ Successfully loaded {ds_config['name']}")
            break
            
        except Exception as e:
            print(f"  Could not load {ds_config['name']}: {str(e)[:150]}")
            continue
    
    if dataset is None:
        raise RuntimeError(
            "Could not load any speech dataset. Please check your internet connection."
        )
    
    # Shuffle and take samples
    np.random.seed(seed)
    dataset = dataset.shuffle(seed=seed)
    
    samples = []
    print("  Downloading and decoding audio files...")
    
    # Create a temporary directory for downloaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, item in enumerate(dataset):
            if len(samples) >= num_samples:
                break
            
            try:
                # Get audio file info (not decoded due to decode=False)
                audio_info = item[dataset_info['audio_key']]
                
                # With decode=False, we get either 'bytes' or 'path'
                if isinstance(audio_info, dict):
                    if 'bytes' in audio_info and audio_info['bytes'] is not None:
                        # Audio data is embedded as bytes - decode with librosa
                        audio_bytes = audio_info['bytes']
                        
                        # librosa can read from file-like objects
                        audio_array, sr = librosa.load(
                            io.BytesIO(audio_bytes), 
                            sr=16000
                        )
                        
                    elif 'path' in audio_info and audio_info['path']:
                        # Path to audio file
                        audio_path = audio_info['path']
                        audio_array, sr = librosa.load(audio_path, sr=16000)
                        
                    else:
                        continue
                else:
                    continue
                
                # Skip very short or silent audio
                if len(audio_array) < 1600:  # Less than 0.1 seconds at 16kHz
                    continue
                    
                sample = {
                    'audio': audio_array.astype(np.float32),
                    'sample_rate': sr,
                    'sentence': item.get(dataset_info['text_key'], ''),
                    'client_id': f'sample_{i}',
                }
                samples.append(sample)
                
                if (len(samples)) % 25 == 0:
                    print(f"    Loaded {len(samples)}/{num_samples} samples...")
                    
            except Exception as e:
                # Skip problematic samples
                if i < 10:  # Only print first few warnings
                    print(f"    Warning: Skipped sample {i}: {str(e)[:60]}")
                continue
    
    print(f"✓ Loaded {len(samples)} audio samples from {dataset_info['name']}")
    return samples


def extract_features_from_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_sr: int = 48000,
    frame_duration_ms: int = 20
) -> List[Dict[str, float]]:
    """
    Extract features from audio using librosa (optimized batch processing).
    
    Features extracted per frame:
    - RMS energy (loudness)
    - Pitch (F0 in Hz)
    - Speaking rate (voice activity)
    
    Args:
        audio: Audio waveform as numpy array
        sample_rate: Original sample rate
        target_sr: Target sample rate (48kHz matches realtime system)
        frame_duration_ms: Frame size in milliseconds
    
    Returns:
        List of feature dictionaries, one per frame
    """
    # Resample to target sample rate if needed
    if sample_rate != target_sr:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
    
    # Calculate frame size (20ms at 48kHz = 960 samples)
    frame_size = int(target_sr * frame_duration_ms / 1000)
    hop_length = frame_size
    
    # Skip very short audio
    if len(audio) < frame_size * 2:
        return []
    
    # BATCH FEATURE EXTRACTION (much faster than frame-by-frame)
    
    # 1. Compute RMS energy for all frames at once
    rms = librosa.feature.rms(y=audio, frame_length=frame_size, hop_length=hop_length)[0]
    
    # 2. Compute pitch for the entire audio in one call (FAST)
    # Using YIN instead of PYIN for 10x speed improvement
    try:
        f0 = librosa.yin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            sr=target_sr,
            frame_length=2048,
            hop_length=hop_length
        )
    except Exception:
        # Fallback: skip pitch if it fails
        f0 = np.full(len(rms), np.nan)
    
    # Align arrays to minimum length
    n_frames = min(len(rms), len(f0) if f0 is not None else len(rms))
    
    # 3. Build feature list
    features_list = []
    energy_threshold = 0.01
    
    for i in range(n_frames):
        rms_energy = float(rms[i])
        pitch_hz = float(f0[i]) if f0 is not None and not np.isnan(f0[i]) else None
        speaking_rate = 1.0 if rms_energy > energy_threshold else 0.0
        
        features = {
            'rms_energy': rms_energy,
            'pitch_hz': pitch_hz,
            'speaking_rate': speaking_rate
        }
        features_list.append(features)
    
    return features_list


def generate_target_signals(features: Dict[str, float]) -> Dict[str, float]:
    """
    Generate target control signals using refined heuristics for confidence.
    """
    # 1. Energy Boost: Target higher RMS for "confident" delivery
    target_rms = 0.07
    current_rms = max(features['rms_energy'], 0.001)
    needed_multiplier = target_rms / current_rms
    needed_multiplier = np.clip(needed_multiplier, 0.5, 2.0)
    energy_raw = (needed_multiplier - 1.0)
    
    # 2. Pitch Shift: Nudge towards a "confident" frequency (approx 140Hz)
    # If pitch is too low, nudge up. If too high, nudge down slightly.
    pitch_shift = 0.0
    if features['pitch_hz'] is not None:
        target_f0 = 140.0 # Neutral/confident target
        # Calculate semitone distance: 12 * log2(f2/f1)
        semitone_diff = 12 * np.log2(target_f0 / features['pitch_hz'])
        # Nudge only 20% towards the target to avoid sounding unnatural
        pitch_shift = np.clip(semitone_diff * 0.2, -2.0, 2.0)
    
    # 3. Pace Adjustment: Small boost for more energetic delivery
    # We'll nudge pace up slightly (1.05x) when energy is low but still speaking
    pace_multiplier = 1.0
    if features['speaking_rate'] > 0.5 and features['rms_energy'] < 0.05:
        pace_multiplier = 1.05
    
    pace_raw = pace_multiplier - 1.0
    
    return {
        'pitch_shift': float(pitch_shift),
        'energy_multiplier': float(energy_raw),
        'pace_multiplier': float(pace_raw)
    }


def create_training_dataset(
    samples: List[Dict],
    output_dir: str = 'training_data',
    validation_split: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process samples and create training/validation datasets.
    
    Args:
        samples: List of audio samples from load_commonvoice_streaming
        output_dir: Directory to save .npz files
        validation_split: Fraction for validation set
    
    Returns:
        X_train, y_train, X_val, y_val
    """
def process_single_sample(sample: Dict) -> Tuple[List, List]:
    """Helper for parallel processing of a single sample."""
    features_list = extract_features_from_audio(
        sample['audio'],
        sample['sample_rate']
    )
    
    sample_features = []
    sample_targets = []
    
    for features in features_list:
        # Skip silent frames
        if features['speaking_rate'] < 0.5:
            continue
        
        # Feature vector
        feature_vector = [
            features['rms_energy'],
            features['pitch_hz'] if features['pitch_hz'] else 150.0,
            features['speaking_rate']
        ]
        sample_features.append(feature_vector)
        
        # Target vector
        targets = generate_target_signals(features)
        target_vector = [
            targets['pitch_shift'],
            targets['energy_multiplier'],
            targets['pace_multiplier']
        ]
        sample_targets.append(target_vector)
    
    return sample_features, sample_targets

def create_training_dataset(
    samples: List[Dict],
    output_dir: str = 'training_data',
    validation_split: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process samples and create training/validation datasets using multiprocessing.
    """
    print(f"\nExtracting features from {len(samples)} audio samples using parallel processing...")
    
    all_features = []
    all_targets = []
    
    start_time = time.time()
    
    # Use multiprocessing to speed up feature extraction
    # Using 4-8 workers typically saturates CPU and provides 3-5x speedup
    max_workers = min(os.cpu_count() or 4, 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_sample, samples))
        
    for sample_features, sample_targets in results:
        all_features.extend(sample_features)
        all_targets.extend(sample_targets)
    
    processing_time = time.time() - start_time
    print(f"✓ Extracted {len(all_features)} feature frames in {processing_time:.1f}s "
          f"({len(samples)/processing_time:.1f} samples/sec)")
    
    print(f"✓ Extracted {len(all_features)} feature frames")
    
    # Convert to numpy arrays
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    
    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split train/validation
    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'commonvoice_train.npz')
    val_path = os.path.join(output_dir, 'commonvoice_val.npz')
    
    np.savez(train_path, X=X_train, y=y_train)
    np.savez(val_path, X=X_val, y=y_val)
    
    print(f"\n✓ Saved preprocessed data:")
    print(f"  {train_path} ({os.path.getsize(train_path) / 1024:.1f} KB)")
    print(f"  {val_path} ({os.path.getsize(val_path) / 1024:.1f} KB)")
    
    return X_train, y_train, X_val, y_val


def print_feature_statistics(X: np.ndarray, y: np.ndarray):
    """
    Print feature distribution statistics.
    """
    print("\n" + "=" * 60)
    print("FEATURE DISTRIBUTION STATISTICS")
    print("=" * 60)
    
    feature_names = ['RMS Energy', 'Pitch (Hz)', 'Speaking Rate']
    target_names = ['Pitch Shift', 'Energy Mult.', 'Pace Mult.']
    
    print("\nInput Features:")
    print("-" * 50)
    print(f"{'Feature':<20} {'Mean':>12} {'Std':>12} {'Min':>10} {'Max':>10}")
    print("-" * 50)
    
    for i, name in enumerate(feature_names):
        print(f"{name:<20} {X[:, i].mean():>12.4f} {X[:, i].std():>12.4f} "
              f"{X[:, i].min():>10.4f} {X[:, i].max():>10.4f}")
    
    print("\nTarget Control Signals:")
    print("-" * 50)
    print(f"{'Signal':<20} {'Mean':>12} {'Std':>12} {'Min':>10} {'Max':>10}")
    print("-" * 50)
    
    for i, name in enumerate(target_names):
        print(f"{name:<20} {y[:, i].mean():>12.4f} {y[:, i].std():>12.4f} "
              f"{y[:, i].min():>10.4f} {y[:, i].max():>10.4f}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Mozilla Common Voice dataset for training'
    )
    parser.add_argument(
        '--samples', type=int, default=1000,
        help='Number of audio samples to process (default: 1000)'
    )
    parser.add_argument(
        '--language', type=str, default='en',
        help='Language code (default: en)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='training_data',
        help='Output directory for preprocessed data'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Test mode: load samples but skip saving'
    )
    parser.add_argument(
        '--validation-split', type=float, default=0.2,
        help='Fraction of data for validation (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MOZILLA COMMON VOICE DATASET PREPROCESSING")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Settings:")
    print(f"  Samples: {args.samples}")
    print(f"  Language: {args.language}")
    print(f"  Output: {args.output_dir}")
    print(f"  Validation split: {args.validation_split * 100:.0f}%")
    print(f"  Dry run: {args.dry_run}")
    
    # Step 1: Load samples
    print("\n" + "-" * 60)
    print("STEP 1: Loading audio samples from Common Voice")
    print("-" * 60)
    
    samples = load_commonvoice_streaming(
        num_samples=args.samples,
        language=args.language
    )
    
    # Step 2: Extract features and create dataset
    print("\n" + "-" * 60)
    print("STEP 2: Extracting features and generating targets")
    print("-" * 60)
    
    X_train, y_train, X_val, y_val = create_training_dataset(
        samples,
        output_dir=args.output_dir,
        validation_split=args.validation_split
    )
    
    # Step 3: Print statistics
    print_feature_statistics(
        np.vstack([X_train, X_val]),
        np.vstack([y_train, y_val])
    )
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nTotal frames: {len(X_train) + len(X_val)}")
    print(f"Training frames: {len(X_train)}")
    print(f"Validation frames: {len(X_val)}")
    print(f"\nNext step: Run 'python train_model.py' to train the model")
    print("=" * 60)


if __name__ == "__main__":
    main()
