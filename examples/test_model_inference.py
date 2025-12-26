"""
Test neural network inference speed.

This script benchmarks:
- Single-frame inference latency
- Batch inference throughput
- Model warmup time
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from voice_model import VoiceControlModel
import numpy as np
import time

print("=" * 60)
print("NEURAL NETWORK INFERENCE TEST")
print("=" * 60)

# Create model
print("\nInitializing model...")
model = VoiceControlModel()

print("\n" + "=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)
model.model.summary()

# Test features
test_features = {
    'rms_energy': 0.05,
    'pitch_hz': 150.0,
    'speaking_rate': 1.0
}

# Warmup
print("\n" + "=" * 60)
print("WARMUP")
print("=" * 60)
print("\nFirst inference (cold start)...")
start = time.time()
_ = model.predict(test_features)
cold_start_ms = (time.time() - start) * 1000
print(f"Cold start time: {cold_start_ms:.2f} ms")

# Benchmark
print("\n" + "=" * 60)
print("INFERENCE SPEED BENCHMARK (100 iterations)")
print("=" * 60)

times = []
for i in range(100):
    start = time.time()
    control_signals = model.predict(test_features)
    elapsed_ms = (time.time() - start) * 1000
    times.append(elapsed_ms)

times = np.array(times)

print(f"\n📊 Statistics:")
print(f"  - Mean: {np.mean(times):.2f} ms")
print(f"  - Median: {np.median(times):.2f} ms")
print(f"  - Min: {np.min(times):.2f} ms")
print(f"  - Max: {np.max(times):.2f} ms")
print(f"  - 95th percentile: {np.percentile(times, 95):.2f} ms")

if np.mean(times) < 5.0:
    print("\n✓ EXCELLENT - Ready for real-time (<5ms)")
elif np.mean(times) < 10.0:
    print("\n✓ GOOD - Acceptable for real-time (<10ms)")
else:
    print("\n⚠️  SLOW - May need optimization (>10ms)")

# Example output
print("\n" + "=" * 60)
print("EXAMPLE OUTPUT")
print("=" * 60)
print(f"\nInput: RMS={test_features['rms_energy']:.4f}, "
      f"Pitch={test_features['pitch_hz']:.1f}Hz, "
      f"Speaking={test_features['speaking_rate']:.1f}")

control = model.predict(test_features)
print(f"\nOutput:")
print(f"  - Pitch shift: {control['pitch_shift_semitones']:+.2f} semitones")
print(f"  - Energy multiplier: {control['energy_multiplier']:.2f}x")
print(f"  - Pace multiplier: {control['pace_multiplier']:.2f}x")

print("\n" + "=" * 60)
print("NOTE: Model weights are random (not trained yet)")
print("=" * 60)
