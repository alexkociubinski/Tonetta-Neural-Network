"""
MODEL EVALUATION SCRIPT

This script evaluates the trained voice model and generates audio examples.

USAGE:
    python evaluate_model.py
    python evaluate_model.py --model trained_model.h5
"""

import numpy as np
import os
import argparse
import json
from datetime import datetime
from typing import Dict, Tuple

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras

# Audio processing
import librosa
import soundfile as sf


def load_model_and_params(model_path: str = 'trained_model.h5'):
    """
    Load trained model and normalization parameters.
    
    Returns:
        model: Keras model
        params: Dictionary with feature_means and feature_stds
    """
    # Load model (compile=False to avoid deserialization issues)
    model = keras.models.load_model(model_path, compile=False)
    print(f"✓ Loaded model from {model_path}")
    
    # Load normalization parameters
    params_path = model_path.replace('.h5', '_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        print(f"✓ Loaded parameters from {params_path}")
    else:
        print("Warning: No parameters file found, using defaults")
        params = {
            'feature_means': [0.05, 150.0, 1.0],
            'feature_stds': [0.05, 50.0, 1.0]
        }
    
    return model, params


def load_validation_data(data_dir: str = 'training_data'):
    """Load validation data from preprocessed files."""
    val_path = os.path.join(data_dir, 'commonvoice_val.npz')
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found at {val_path}")
    
    data = np.load(val_path)
    return data['X'], data['y']


def evaluate_model(model, params, X_val, y_val):
    """
    Evaluate model and compute per-output metrics.
    
    Returns:
        Dictionary with metrics for each output
    """
    # Normalize inputs
    means = np.array(params['feature_means'])
    stds = np.array(params['feature_stds'])
    stds = np.where(stds == 0, 1.0, stds)  # Avoid division by zero
    
    X_val_norm = (X_val - means) / stds
    X_val_norm = np.nan_to_num(X_val_norm, nan=0.0)
    
    # Get predictions
    predictions = model.predict(X_val_norm, verbose=0)
    
    # Compute metrics
    output_names = ['pitch_shift', 'energy_multiplier', 'pace_multiplier']
    metrics = {}
    
    print("\n" + "=" * 60)
    print("PER-OUTPUT METRICS")
    print("=" * 60)
    print(f"\n{'Output':<25} {'MSE':<15} {'MAE':<15} {'RMSE':<15}")
    print("-" * 70)
    
    for i, name in enumerate(output_names):
        mse = np.mean((predictions[:, i] - y_val[:, i]) ** 2)
        mae = np.mean(np.abs(predictions[:, i] - y_val[:, i]))
        rmse = np.sqrt(mse)
        
        metrics[name] = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse)
        }
        
        print(f"{name:<25} {mse:<15.6f} {mae:<15.6f} {rmse:<15.6f}")
    
    # Overall metrics
    overall_mse = np.mean((predictions - y_val) ** 2)
    overall_mae = np.mean(np.abs(predictions - y_val))
    
    print("-" * 70)
    print(f"{'OVERALL':<25} {overall_mse:<15.6f} {overall_mae:<15.6f}")
    
    metrics['overall'] = {
        'mse': float(overall_mse),
        'mae': float(overall_mae)
    }
    
    return metrics, predictions


def generate_audio_examples(
    model, 
    params,
    num_examples: int = 3,
    output_dir: str = 'examples/evaluation'
):
    """
    Generate before/after audio examples showing voice modifications.
    
    This loads sample audio, runs through the model, and applies modifications.
    """
    from audio_modifier import AudioModifier
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("GENERATING AUDIO EXAMPLES")
    print("=" * 60)
    
    # Load some audio samples from the validation data
    # We need to regenerate audio since we only saved features
    # For now, generate synthetic test tones
    
    sample_rate = 48000
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create audio modifier
    modifier = AudioModifier(sample_rate)
    
    # Normalization params
    means = np.array(params['feature_means'])
    stds = np.array(params['feature_stds'])
    stds = np.where(stds == 0, 1.0, stds)
    
    # Generate 3 different test signals
    test_signals = [
        {
            'name': 'low_energy',
            'audio': 0.02 * np.sin(2 * np.pi * 200 * t).astype(np.float32),
            'description': 'Low energy (quiet) 200Hz tone'
        },
        {
            'name': 'medium_energy',
            'audio': 0.05 * np.sin(2 * np.pi * 150 * t).astype(np.float32),
            'description': 'Medium energy 150Hz tone'
        },
        {
            'name': 'high_energy',
            'audio': 0.1 * np.sin(2 * np.pi * 300 * t).astype(np.float32),
            'description': 'High energy (loud) 300Hz tone'
        }
    ]
    
    for i, signal in enumerate(test_signals[:num_examples]):
        print(f"\n  Example {i+1}: {signal['description']}")
        
        audio = signal['audio']
        
        # Extract features (simplified - just use global stats)
        rms = np.sqrt(np.mean(audio ** 2))
        pitch_hz = 150.0  # Approximate
        speaking_rate = 1.0
        
        # Create feature vector and normalize
        features = np.array([[rms, pitch_hz, speaking_rate]])
        features_norm = (features - means) / stds
        features_norm = np.nan_to_num(features_norm, nan=0.0)
        
        # Get model predictions
        control_signals = model.predict(features_norm, verbose=0)[0]
        
        # Scale outputs to actual ranges (from VoiceControlModel)
        pitch_shift_range = 5.0
        energy_boost_range = 1.0
        pace_adjustment_range = 0.2
        
        pitch_shift = control_signals[0] * pitch_shift_range
        energy_mult = 1.0 + (control_signals[1] * energy_boost_range)
        pace_mult = 1.0 + (control_signals[2] * pace_adjustment_range)
        
        print(f"    Input RMS: {rms:.4f}")
        print(f"    Predicted control signals:")
        print(f"      - Pitch shift: {pitch_shift:+.2f} semitones")
        print(f"      - Energy mult: {energy_mult:.2f}x")
        print(f"      - Pace mult: {pace_mult:.2f}x")
        
        # Save before audio
        before_path = os.path.join(output_dir, f"{signal['name']}_before.wav")
        sf.write(before_path, audio, sample_rate)
        
        # Apply modifications
        try:
            modified_audio, _ = modifier.apply_modifications(
                audio,
                pitch_shift_semitones=pitch_shift,
                energy_multiplier=energy_mult,
                pace_multiplier=pace_mult
            )
            
            # Save after audio
            after_path = os.path.join(output_dir, f"{signal['name']}_after.wav")
            sf.write(after_path, modified_audio, sample_rate)
            
            print(f"    ✓ Saved: {before_path}")
            print(f"    ✓ Saved: {after_path}")
            
        except Exception as e:
            print(f"    ⚠ Could not apply modifications: {e}")
            print(f"    ✓ Saved before only: {before_path}")
    
    print(f"\n✓ Audio examples saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained voice model'
    )
    parser.add_argument(
        '--model', type=str, default='trained_model.h5',
        help='Path to trained model file'
    )
    parser.add_argument(
        '--data-dir', type=str, default='training_data',
        help='Directory containing validation data'
    )
    parser.add_argument(
        '--output-dir', type=str, default='examples/evaluation',
        help='Directory for audio examples'
    )
    parser.add_argument(
        '--num-examples', type=int, default=3,
        help='Number of audio examples to generate'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VOICE MODEL EVALUATION")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model
    print("\n" + "-" * 60)
    print("LOADING MODEL")
    print("-" * 60)
    model, params = load_model_and_params(args.model)
    
    # Load validation data
    print("\n" + "-" * 60)
    print("LOADING VALIDATION DATA")
    print("-" * 60)
    X_val, y_val = load_validation_data(args.data_dir)
    print(f"  Validation samples: {len(X_val)}")
    
    # Evaluate
    print("\n" + "-" * 60)
    print("EVALUATING MODEL")
    print("-" * 60)
    metrics, predictions = evaluate_model(model, params, X_val, y_val)
    
    # Generate audio examples
    print("\n" + "-" * 60)
    print("GENERATING EXAMPLES")
    print("-" * 60)
    generate_audio_examples(
        model, params, 
        num_examples=args.num_examples,
        output_dir=args.output_dir
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nOverall MSE: {metrics['overall']['mse']:.6f}")
    print(f"Overall MAE: {metrics['overall']['mae']:.6f}")
    print(f"\nAudio examples: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
