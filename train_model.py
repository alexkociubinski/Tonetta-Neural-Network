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
from typing import List, Dict, Tuple
from datetime import datetime

from voice_model import VoiceControlModel
from audio_pipeline import AudioFeatureExtractor


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
        
        X_train_norm = (X_train - self.model.feature_means) / self.model.feature_stds
        X_val_norm = (X_val - self.model.feature_means) / self.model.feature_stds
        
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


# Example usage and demo
if __name__ == "__main__":
    print("=" * 60)
    print("MODEL TRAINING DEMO")
    print("=" * 60)
    print("\nThis demonstrates how to train the neural network.")
    print("For a real startup, you'd collect actual call recordings.\n")
    
    # Step 1: Create synthetic training data (for demo)
    print("Step 1: Creating synthetic training data...")
    print("(In production, you'd use real recorded calls)\n")
    
    # Generate synthetic examples
    np.random.seed(42)
    training_examples = []
    
    for i in range(100):  # 100 synthetic examples
        # Random features
        features_list = []
        for j in range(50):  # 50 frames per example
            features = {
                'rms_energy': np.random.uniform(0.01, 0.1),
                'pitch_hz': np.random.uniform(80, 300),
                'speaking_rate': np.random.uniform(0, 1),
                'processing_time_ms': 5.0
            }
            features_list.append(features)
        
        # Random confidence level
        confidence = np.random.uniform(0.3, 0.9)
        
        example = {
            'audio_file': f'synthetic_{i}.wav',
            'features': features_list,
            'targets': {
                'pitch_shift': (confidence - 0.5) * 4.0,
                'energy_multiplier': 0.8 + (confidence * 0.4),
                'pace_multiplier': 1.0
            },
            'labels': {
                'confidence': confidence,
                'energy': 1.0,
                'pace': 1.0
            }
        }
        training_examples.append(example)
    
    print(f"✓ Created {len(training_examples)} synthetic examples")
    
    # Step 2: Prepare dataset
    print("\nStep 2: Preparing dataset...")
    trainer = VoiceModelTrainer()
    X_train, y_train, X_val, y_val = trainer.prepare_dataset(training_examples)
    
    # Step 3: Train model
    print("\nStep 3: Training model...")
    trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,  # Fewer epochs for demo
        batch_size=32,
        learning_rate=0.001
    )
    
    # Step 4: Evaluate
    print("\nStep 4: Evaluating model...")
    trainer.evaluate(X_val, y_val)
    
    # Step 5: Save
    print("\nStep 5: Saving trained model...")
    trainer.save_model('models/demo_trained_model.h5')
    
    print("\n" + "=" * 60)
    print("NEXT STEPS FOR REAL TRAINING")
    print("=" * 60)
    print("\n1. Collect real audio data:")
    print("   - Record practice calls")
    print("   - Label with confidence ratings (1-10)")
    print("   - Store as WAV files")
    print("\n2. Use TrainingDataCollector:")
    print("   - Extract features from recordings")
    print("   - Create training examples")
    print("\n3. Train on real data:")
    print("   - Use VoiceModelTrainer")
    print("   - Monitor validation loss")
    print("   - Save best model")
    print("\n4. Deploy trained model:")
    print("   - Load in realtime_system.py")
    print("   - Test on live audio")
    print("   - Collect user feedback")
    print("\n" + "=" * 60)
