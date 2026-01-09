"""
STEP 2: NEURAL NETWORK (ANALYSIS-ONLY)

This module defines a lightweight neural network that learns to control voice modifications.

PLAIN ENGLISH EXPLANATION - WHAT IS A NEURAL NETWORK?

Imagine you're teaching a student to adjust radio knobs (pitch, volume, speed) based on 
what they hear. At first, they're terrible - they turn knobs randomly. But after seeing 
thousands of examples of "when you hear THIS, turn the knobs THAT way," they start to 
recognize patterns.

A neural network is exactly this: a mathematical student that learns from examples.

HOW IT WORKS:
1. You feed it numbers (audio features: energy, pitch, speaking rate)
2. It does math on those numbers using "weights" (knob positions it learned)
3. It outputs new numbers (how much to shift pitch, boost energy, adjust pace)
4. During training, we adjust the weights to make better predictions

WHY THIS ARCHITECTURE:
- Small enough to run in <5ms on a laptop CPU
- Big enough to learn useful patterns
- Designed for streaming (processes one frame at a time)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
from typing import Dict, Tuple


class VoiceControlModel:
    """
    Neural network that generates voice modification control signals.
    
    ARCHITECTURE EXPLAINED:
    
    Input Layer (3 neurons):
        - Takes in: [rms_energy, pitch_hz, speaking_rate]
        - These are the "sensors" from Step 1
    
    Hidden Layer 1 (32 neurons):
        - Each neuron computes: output = ReLU(weights · input + bias)
        - ReLU = "Rectified Linear Unit" = max(0, x)
        - Why? Adds non-linearity so network can learn curves, not just straight lines
        - Think of each neuron as a "feature detector" learning patterns
    
    Hidden Layer 2 (16 neurons):
        - Processes the 32 features from Layer 1
        - Combines patterns into higher-level concepts
        - Example: "high energy + high pitch + speaking = excited voice"
    
    Output Layer (3 neurons):
        - Produces: [pitch_shift, energy_boost, pace_adjustment]
        - Uses tanh activation to constrain outputs to [-1, 1] range
        - We'll scale these to useful ranges (e.g., ±5 semitones for pitch)
    
    TOTAL PARAMETERS: ~2,000 (very small!)
    - Layer 1: 3 inputs × 32 neurons + 32 biases = 128 parameters
    - Layer 2: 32 inputs × 16 neurons + 16 biases = 528 parameters
    - Output: 16 inputs × 3 neurons + 3 biases = 51 parameters
    """
    
    def __init__(self, input_size: int = 3, hidden_sizes: Tuple[int, int] = (32, 16)):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features (3: energy, pitch, speaking_rate)
            hidden_sizes: Sizes of hidden layers (32, 16 for speed)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        # Build the model
        self.model = self._build_model()
        
        # Feature normalization parameters (will be set during training)
        # For now, use reasonable defaults
        self.feature_means = np.array([0.05, 150.0, 0.5])  # RMS, pitch, speaking
        self.feature_stds = np.array([0.05, 50.0, 0.5])
        
        # Output scaling parameters
        self.pitch_shift_range = 5.0  # ±5 semitones
        self.energy_boost_range = 1.0  # ±1.0 (0.0 to 2.0 multiplier)
        self.pace_adjustment_range = 0.2  # ±0.2 (0.8x to 1.2x speed)
        
        print("✓ Neural network initialized:")
        print(f"  - Architecture: {input_size} → {hidden_sizes[0]} → {hidden_sizes[1]} → 3")
        print(f"  - Total parameters: {self.model.count_params():,}")
        print(f"  - Output ranges:")
        print(f"    • Pitch shift: ±{self.pitch_shift_range} semitones")
        print(f"    • Energy boost: 0.0 to 2.0x")
        print(f"    • Pace adjustment: 0.8x to 1.2x")
    
    def _build_model(self) -> keras.Model:
        """
        Build the neural network architecture.
        
        LAYER-BY-LAYER EXPLANATION:
        
        1. Input Layer:
           - Just defines the shape: (batch_size, 3)
           - No computation happens here
        
        2. Dense Layer 1 (32 neurons):
           - "Dense" = every input connects to every neuron (fully connected)
           - Each neuron: output = ReLU(w1*energy + w2*pitch + w3*speaking + bias)
           - ReLU(x) = max(0, x) - kills negative values, keeps positive
           - Why ReLU? Fast to compute, prevents "vanishing gradients" during training
        
        3. Dense Layer 2 (16 neurons):
           - Takes 32 inputs from Layer 1
           - Combines patterns: "if neurons 5, 12, 27 are active, then..."
           - Still uses ReLU activation
        
        4. Output Layer (3 neurons):
           - Uses tanh activation: output = (e^x - e^-x) / (e^x + e^-x)
           - tanh squashes outputs to [-1, 1] range
           - We'll scale these to meaningful ranges later
        
        Returns:
            Compiled Keras model ready for inference (or training later)
        """
        # Sequential model = layers stacked one after another
        model = keras.Sequential([
            # Input layer - defines shape
            layers.Input(shape=(self.input_size,), name='audio_features'),
            
            # Hidden layer 1 - learns basic patterns
            layers.Dense(
                self.hidden_sizes[0],
                activation='relu',
                name='hidden_1',
                kernel_initializer='he_normal'  # Good initialization for ReLU
            ),
            
            # Hidden layer 2 - combines patterns
            layers.Dense(
                self.hidden_sizes[1],
                activation='relu',
                name='hidden_2',
                kernel_initializer='he_normal'
            ),
            
            # Output layer - produces control signals
            layers.Dense(
                3,  # pitch_shift, energy_boost, pace_adjustment
                activation='tanh',  # Outputs in [-1, 1]
                name='control_outputs'
            )
        ], name='VoiceControlModel')
        
        # Compile model (needed even for inference-only)
        model.compile(
            optimizer='adam',  # Will be used if we train later
            loss='mse'  # Mean squared error
        )
        
        return model
    
    def normalize_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Normalize input features to zero mean, unit variance.
        
        PLAIN ENGLISH:
        - Neural networks work best when inputs are similar scales
        - Energy might be 0.01-0.1, pitch might be 80-300 Hz - very different!
        - Normalization: (value - mean) / std_dev
        - Result: all features centered around 0, similar ranges
        
        Args:
            features: Dictionary with 'rms_energy', 'pitch_hz', 'speaking_rate'
        
        Returns:
            Normalized numpy array [energy, pitch, speaking]
        """
        # Extract features in correct order
        raw_features = np.array([
            features['rms_energy'],
            features['pitch_hz'] if features['pitch_hz'] is not None else 150.0,  # Default pitch
            features['speaking_rate']
        ])
        
        # Normalize: (x - mean) / std
        normalized = (raw_features - self.feature_means) / self.feature_stds
        
        return normalized
    
    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Run inference to get control signals for voice modification.
        
        DATA FLOW:
        1. Raw features → Normalize → [3 numbers]
        2. Layer 1: [3] → [32] via matrix multiplication + ReLU
        3. Layer 2: [32] → [16] via matrix multiplication + ReLU
        4. Output: [16] → [3] via matrix multiplication + tanh
        5. Scale outputs to useful ranges
        
        Args:
            features: Dictionary from audio_pipeline.py
        
        Returns:
            Dictionary with 'pitch_shift_semitones', 'energy_multiplier', 'pace_multiplier'
        """
        start_time = time.time()
        
        # Step 1: Normalize inputs
        normalized_features = self.normalize_features(features)
        
        # Step 2: Add batch dimension (model expects shape: [batch_size, features])
        input_batch = normalized_features.reshape(1, -1)
        
        # Step 3: Run neural network inference
        # This is where the magic happens - all the learned weights are applied
        raw_outputs = self.model.predict(input_batch, verbose=0)[0]  # Shape: (3,)
        
        # Step 4: Scale outputs from [-1, 1] to useful ranges
        control_signals = {
            'pitch_shift_semitones': raw_outputs[0] * self.pitch_shift_range,
            'energy_multiplier': 1.0 + (raw_outputs[1] * self.energy_boost_range),
            'pace_multiplier': 1.0 + (raw_outputs[2] * self.pace_adjustment_range),
            'inference_time_ms': (time.time() - start_time) * 1000
        }
        
        return control_signals
    
    def predict_batch(self, features_list: list) -> np.ndarray:
        """
        Run inference on multiple frames at once (more efficient).
        
        PLAIN ENGLISH:
        - Processing 10 frames together is faster than 10 separate predictions
        - GPUs/CPUs can do parallel math on batches
        - Useful when you have a backlog of frames to process
        
        Args:
            features_list: List of feature dictionaries
        
        Returns:
            Array of control signals, shape (batch_size, 3)
        """
        # Normalize all features
        normalized_batch = np.array([
            self.normalize_features(features) for features in features_list
        ])
        
        # Run inference on entire batch
        raw_outputs = self.model.predict(normalized_batch, verbose=0)
        
        # Scale outputs
        scaled_outputs = np.zeros_like(raw_outputs)
        scaled_outputs[:, 0] = raw_outputs[:, 0] * self.pitch_shift_range
        scaled_outputs[:, 1] = 1.0 + (raw_outputs[:, 1] * self.energy_boost_range)
        scaled_outputs[:, 2] = 1.0 + (raw_outputs[:, 2] * self.pace_adjustment_range)
        
        return scaled_outputs
    
    def save(self, filepath: str):
        """Save model weights to disk."""
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights from disk."""
        # Use compile=False to avoid issues with custom metrics during loading
        self.model = keras.models.load_model(filepath, compile=False)
        print(f"✓ Model loaded from {filepath}")


# Demo: Run this file directly to test the neural network
if __name__ == "__main__":
    print("=" * 60)
    print("TONETTA NEURAL NETWORK - STEP 2 DEMO")
    print("=" * 60)
    print("\nThis demonstrates the neural network architecture and inference speed.\n")
    
    # Create model
    model = VoiceControlModel()
    
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    model.model.summary()
    
    # Test inference speed
    print("\n" + "=" * 60)
    print("INFERENCE SPEED TEST")
    print("=" * 60)
    
    # Create dummy features
    test_features = {
        'rms_energy': 0.05,
        'pitch_hz': 150.0,
        'speaking_rate': 1.0
    }
    
    # Warm up (first inference is slower due to TensorFlow initialization)
    print("\nWarming up model...")
    _ = model.predict(test_features)
    
    # Benchmark inference speed
    print("Running 100 inferences...\n")
    times = []
    
    for i in range(100):
        start = time.time()
        control_signals = model.predict(test_features)
        elapsed_ms = (time.time() - start) * 1000
        times.append(elapsed_ms)
    
    # Statistics
    times = np.array(times)
    print(f"📊 Inference Speed Statistics:")
    print(f"  - Mean: {np.mean(times):.2f} ms")
    print(f"  - Median: {np.median(times):.2f} ms")
    print(f"  - Min: {np.min(times):.2f} ms")
    print(f"  - Max: {np.max(times):.2f} ms")
    print(f"  - 95th percentile: {np.percentile(times, 95):.2f} ms")
    
    if np.mean(times) < 5.0:
        print("\n✓ Inference speed excellent (<5ms) - ready for real-time!")
    elif np.mean(times) < 10.0:
        print("\n✓ Inference speed good (<10ms) - acceptable for real-time")
    else:
        print("\n⚠️  Inference speed slow (>10ms) - may need optimization")
    
    # Show example output
    print("\n" + "=" * 60)
    print("EXAMPLE CONTROL SIGNALS")
    print("=" * 60)
    print(f"\nInput features:")
    print(f"  - RMS Energy: {test_features['rms_energy']:.4f}")
    print(f"  - Pitch: {test_features['pitch_hz']:.1f} Hz")
    print(f"  - Speaking Rate: {test_features['speaking_rate']:.1f}")
    
    control_signals = model.predict(test_features)
    print(f"\nOutput control signals:")
    print(f"  - Pitch shift: {control_signals['pitch_shift_semitones']:+.2f} semitones")
    print(f"  - Energy multiplier: {control_signals['energy_multiplier']:.2f}x")
    print(f"  - Pace multiplier: {control_signals['pace_multiplier']:.2f}x")
    
    print("\n" + "=" * 60)
    print("NOTE: Model weights are currently random (not trained).")
    print("In production, these weights will be learned from training data.")
    print("=" * 60)
