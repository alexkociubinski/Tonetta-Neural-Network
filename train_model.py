"""
MODEL TRAINING SYSTEM

This module provides everything you need to train the neural network on real data.

TRAINING WORKFLOW:
1. Collect data: Record calls with confidence ratings
2. Prepare data: Extract features and create training dataset
3. Train model: Learn weights that map features → control signals
4. Evaluate: Test on validation set
5. Deploy: Save trained model for real-time use

PLAIN ENGLISH:
Right now, the neural network has random weights (like a student who hasn't studied).
Training teaches it the right patterns by showing it examples:
"When you hear THIS voice (features), apply THESE modifications (control signals)"
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import json
import os
import argparse
from typing import List, Dict, Tuple
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from voice_model import VoiceControlModel
from audio_pipeline import AudioFeatureExtractor


def load_preprocessed_data(data_dir: str = 'training_data') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed .npz files from disk.
    
    Args:
        data_dir: Directory containing commonvoice_train.npz and commonvoice_val.npz
    
    Returns:
        X_train, y_train, X_val, y_val
    """
    train_path = os.path.join(data_dir, 'commonvoice_train.npz')
    val_path = os.path.join(data_dir, 'commonvoice_val.npz')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Run preprocess_commonvoice.py first."
        )
    
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    
    X_train = train_data['X']
    y_train = train_data['y']
    X_val = val_data['X']
    y_val = val_data['y']
    
    print(f"Loaded preprocessed data from {data_dir}:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    return X_train, y_train, X_val, y_val


def save_training_log(history, output_path: str = 'training_log.txt'):
    """
    Save training history and loss curves.
    
    Args:
        history: Keras training history object
        output_path: Path to save the log
    """
    # Save loss curves as image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # MAE curve
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.set_title('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.txt', '.png'), dpi=150)
    plt.close()
    
    # Save text log
    with open(output_path, 'w') as f:
        f.write("TRAINING LOG\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Final Metrics:\n")
        f.write(f"  Training Loss: {history.history['loss'][-1]:.6f}\n")
        f.write(f"  Validation Loss: {history.history['val_loss'][-1]:.6f}\n")
        f.write(f"  Training MAE: {history.history['mae'][-1]:.6f}\n")
        f.write(f"  Validation MAE: {history.history['val_mae'][-1]:.6f}\n")
        f.write(f"  Epochs completed: {len(history.history['loss'])}\n\n")
        
        f.write("Epoch-by-Epoch:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Train MAE':<15} {'Val MAE':<15}\n")
        f.write("-" * 60 + "\n")
        
        for i in range(len(history.history['loss'])):
            f.write(f"{i+1:<8} {history.history['loss'][i]:<15.6f} "
                   f"{history.history['val_loss'][i]:<15.6f} "
                   f"{history.history['mae'][i]:<15.6f} "
                   f"{history.history['val_mae'][i]:<15.6f}\n")
    
    print(f"\n✓ Training log saved to {output_path}")
    print(f"✓ Loss curves saved to {output_path.replace('.txt', '.png')}")


class TrainingDataCollector:
    """
    Collects training data from recorded audio with labels.
    
    WHAT YOU NEED:
    - Audio recordings of calls
    - Labels for each recording (desired confidence level, energy, etc.)
    - This class extracts features and creates training dataset
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.feature_extractor = AudioFeatureExtractor(sample_rate)
        
    def extract_features_from_audio(self, audio_file: str) -> List[Dict]:
        """
        Extract features from an audio file.
        
        Args:
            audio_file: Path to WAV file
        
        Returns:
            List of feature dictionaries (one per frame)
        """
        import librosa
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=self.sample_rate)
        
        # Process in frames
        frame_size = self.feature_extractor.frame_size
        features_list = []
        
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            features = self.feature_extractor.extract_features(frame)
            features_list.append(features)
        
        return features_list
    
    def create_training_example(
        self,
        audio_file: str,
        target_confidence: float,  # 0.0 to 1.0
        target_energy: float = 1.0,  # 0.5 to 2.0
        target_pace: float = 1.0  # 0.8 to 1.2
    ) -> Dict:
        """
        Create a training example from audio + labels.
        
        PLAIN ENGLISH:
        - Extract features from the audio
        - Pair them with desired modifications
        - This becomes one training example
        
        Args:
            audio_file: Path to recorded audio
            target_confidence: Desired confidence level (0-1)
            target_energy: Desired energy multiplier
            target_pace: Desired pace multiplier
        
        Returns:
            Dictionary with features and targets
        """
        features_list = self.extract_features_from_audio(audio_file)
        
        # Convert confidence to control signals
        # Higher confidence → slight pitch increase, energy boost
        pitch_shift = (target_confidence - 0.5) * 4.0  # -2 to +2 semitones
        energy_mult = 0.8 + (target_confidence * 0.4)  # 0.8 to 1.2x
        
        return {
            'audio_file': audio_file,
            'features': features_list,
            'targets': {
                'pitch_shift': pitch_shift,
                'energy_multiplier': energy_mult,
                'pace_multiplier': target_pace
            },
            'labels': {
                'confidence': target_confidence,
                'energy': target_energy,
                'pace': target_pace
            }
        }


class VoiceModelTrainer:
    """
    Trains the neural network on collected data.
    
    TRAINING PROCESS:
    1. Load training data (features + targets)
    2. Split into train/validation sets
    3. Train model (adjust weights to minimize error)
    4. Evaluate on validation set
    5. Save trained model
    """
    
    def __init__(self, model: VoiceControlModel = None):
        """
        Initialize trainer.
        
        Args:
            model: VoiceControlModel to train (creates new if None)
        """
        self.model = model if model else VoiceControlModel()
        self.history = None
        
    def prepare_dataset(
        self,
        training_examples: List[Dict],
        validation_split: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and validation datasets.
        
        PLAIN ENGLISH:
        - Collect all features and targets from examples
        - Shuffle the data (prevents learning order)
        - Split into train (80%) and validation (20%)
        - Validation set tests if model generalizes to new data
        
        Args:
            training_examples: List of examples from TrainingDataCollector
            validation_split: Fraction to use for validation
        
        Returns:
            X_train, y_train, X_val, y_val
        """
        all_features = []
        all_targets = []
        
        # Collect all features and targets
        for example in training_examples:
            for features in example['features']:
                # Extract input features
                feature_vector = [
                    features['rms_energy'],
                    features['pitch_hz'] if features['pitch_hz'] else 150.0,
                    features['speaking_rate']
                ]
                all_features.append(feature_vector)
                
                # Extract target control signals
                target_vector = [
                    example['targets']['pitch_shift'],
                    example['targets']['energy_multiplier'],
                    example['targets']['pace_multiplier']
                ]
                all_targets.append(target_vector)
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_targets)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Split train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Dataset prepared:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        
        return X_train, y_train, X_val, y_val
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Train the neural network.
        
        WHAT HAPPENS DURING TRAINING:
        1. Model makes predictions on training data
        2. Calculate error (how wrong the predictions are)
        3. Adjust weights to reduce error (backpropagation)
        4. Repeat for many epochs (passes through data)
        5. Check validation error to prevent overfitting
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: How many times to go through the data
            batch_size: How many examples to process at once
            learning_rate: How fast to adjust weights
        """
        print("\n" + "=" * 60)
        print("TRAINING NEURAL NETWORK")
        print("=" * 60)
        
        # Normalize features
        self.model.feature_means = np.mean(X_train, axis=0)
        self.model.feature_stds = np.std(X_train, axis=0)
        
        # Handle columns with zero std (e.g., speaking_rate is always 1.0)
        # Replace zero std with 1.0 to avoid division by zero
        self.model.feature_stds = np.where(
            self.model.feature_stds == 0, 
            1.0, 
            self.model.feature_stds
        )
        
        X_train_norm = (X_train - self.model.feature_means) / self.model.feature_stds
        X_val_norm = (X_val - self.model.feature_means) / self.model.feature_stds
        
        # Check for NaN values
        if np.isnan(X_train_norm).any():
            print("Warning: NaN values detected in training data, replacing with 0")
            X_train_norm = np.nan_to_num(X_train_norm, nan=0.0)
        if np.isnan(X_val_norm).any():
            X_val_norm = np.nan_to_num(X_val_norm, nan=0.0)
        
        # Compile model with optimizer
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.model.compile(
            optimizer=optimizer,
            loss='mse',  # Mean squared error
            metrics=['mae']  # Mean absolute error
        )
        
        # Callbacks for training
        callback_list = [
            # Save best model
            callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # Stop if not improving
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate if stuck
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1
            )
        ]
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Train!
        print(f"\nTraining for {epochs} epochs...")
        self.history = self.model.model.fit(
            X_train_norm, y_train,
            validation_data=(X_val_norm, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        print("\n✓ Training complete!")
        
    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluate model on validation set."""
        X_val_norm = (X_val - self.model.feature_means) / self.model.feature_stds
        X_val_norm = np.nan_to_num(X_val_norm, nan=0.0)
        
        loss, mae = self.model.model.evaluate(X_val_norm, y_val, verbose=0)
        
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        print(f"  Loss (MSE): {loss:.4f}")
        print(f"  Mean Absolute Error: {mae:.4f}")
        
        # Show example predictions
        print("\nExample predictions:")
        for i in range(min(5, len(X_val))):
            pred = self.model.model.predict(X_val_norm[i:i+1], verbose=0)[0]
            true = y_val[i]
            
            print(f"\n  Example {i+1}:")
            print(f"    Predicted: pitch={pred[0]:+.2f}, energy={pred[1]:.2f}, pace={pred[2]:.2f}")
            print(f"    True:      pitch={true[0]:+.2f}, energy={true[1]:.2f}, pace={true[2]:.2f}")
    
    def save_model(self, filepath: str = 'models/trained_model.h5'):
        """Save trained model to disk."""
        self.model.save(filepath)
        
        # Save normalization parameters
        params = {
            'feature_means': self.model.feature_means.tolist(),
            'feature_stds': self.model.feature_stds.tolist()
        }
        
        with open(filepath.replace('.h5', '_params.json'), 'w') as f:
            json.dump(params, f)
        
        print(f"\n✓ Model saved to {filepath}")


# Command-line interface for training
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train voice model on preprocessed Common Voice data'
    )
    parser.add_argument(
        '--data-dir', type=str, default='training_data',
        help='Directory containing preprocessed .npz files'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.001,
        help='Learning rate for Adam optimizer (default: 0.001)'
    )
    parser.add_argument(
        '--patience', type=int, default=10,
        help='Early stopping patience (default: 10)'
    )
    parser.add_argument(
        '--output', type=str, default='trained_model.h5',
        help='Output path for trained model (default: trained_model.h5)'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run with synthetic data (demo mode)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VOICE MODEL TRAINING")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Settings:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Output model: {args.output}")
    
    if args.demo:
        # Demo mode with synthetic data
        print("\n" + "-" * 60)
        print("DEMO MODE: Using synthetic data")
        print("-" * 60)
        
        np.random.seed(42)
        training_examples = []
        
        for i in range(100):
            features_list = []
            for j in range(50):
                features = {
                    'rms_energy': np.random.uniform(0.01, 0.1),
                    'pitch_hz': np.random.uniform(80, 300),
                    'speaking_rate': np.random.uniform(0, 1),
                }
                features_list.append(features)
            
            confidence = np.random.uniform(0.3, 0.9)
            
            example = {
                'features': features_list,
                'targets': {
                    'pitch_shift': (confidence - 0.5) * 4.0,
                    'energy_multiplier': 0.8 + (confidence * 0.4),
                    'pace_multiplier': 1.0
                }
            }
            training_examples.append(example)
        
        trainer = VoiceModelTrainer()
        X_train, y_train, X_val, y_val = trainer.prepare_dataset(training_examples)
        
    else:
        # Production mode with preprocessed Common Voice data
        print("\n" + "-" * 60)
        print("LOADING PREPROCESSED DATA")
        print("-" * 60)
        
        X_train, y_train, X_val, y_val = load_preprocessed_data(args.data_dir)
        trainer = VoiceModelTrainer()
    
    # Train model
    print("\n" + "-" * 60)
    print("TRAINING")
    print("-" * 60)
    
    trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Evaluate
    print("\n" + "-" * 60)
    print("EVALUATION")
    print("-" * 60)
    trainer.evaluate(X_val, y_val)
    
    # Save model
    print("\n" + "-" * 60)
    print("SAVING MODEL")
    print("-" * 60)
    trainer.save_model(args.output)
    
    # Save training log
    if trainer.history:
        save_training_log(trainer.history, 'training_log.txt')
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel saved to: {args.output}")
    print(f"Training log: training_log.txt")
    print(f"Loss curves: training_log.png")
    print("\nNext steps:")
    print("  1. Run 'python evaluate_model.py' to test the model")
    print("  2. Load in realtime_system.py for live voice modification")
    print("=" * 60)
