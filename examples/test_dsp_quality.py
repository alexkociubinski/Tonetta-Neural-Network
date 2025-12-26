"""
Test DSP audio modification quality and speed.

This script tests:
- Pitch shifting
- Energy adjustment
- Pace modification
- Combined modifications
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from audio_modifier import AudioModifier
import numpy as np
import time

print("=" * 60)
print("DSP MODIFICATION TEST")
print("=" * 60)

# Create modifier
modifier = AudioModifier(sample_rate=48000)

# Generate test audio (440 Hz sine wave = musical note A)
print("\nGenerating test audio (440 Hz sine wave)...")
duration = 1.0
t = np.linspace(0, duration, int(48000 * duration))
test_audio = 0.3 * np.sin(2 * np.pi * 440 * t)

# Test 1: Pitch shifting
print("\n" + "=" * 60)
print("TEST 1: PITCH SHIFTING")
print("=" * 60)

for semitones in [-12, -5, 0, +5, +12]:
    start = time.time()
    shifted = modifier.shift_pitch(test_audio, semitones)
    elapsed_ms = (time.time() - start) * 1000
    print(f"  Shift {semitones:+3d} semitones: {elapsed_ms:6.2f} ms")

# Test 2: Energy adjustment
print("\n" + "=" * 60)
print("TEST 2: ENERGY ADJUSTMENT")
print("=" * 60)

for mult in [0.5, 0.75, 1.0, 1.5, 2.0]:
    start = time.time()
    adjusted = modifier.adjust_energy(test_audio, mult)
    elapsed_ms = (time.time() - start) * 1000
    print(f"  Multiplier {mult:.2f}x: {elapsed_ms:6.2f} ms")

# Test 3: Pace adjustment
print("\n" + "=" * 60)
print("TEST 3: PACE ADJUSTMENT")
print("=" * 60)

for rate in [0.8, 0.9, 1.0, 1.1, 1.2]:
    start = time.time()
    stretched = modifier.adjust_pace(test_audio, rate)
    elapsed_ms = (time.time() - start) * 1000
    output_duration = len(stretched) / 48000
    print(f"  Rate {rate:.1f}x: {elapsed_ms:6.2f} ms (output: {output_duration:.2f}s)")

# Test 4: Real-time frame processing
print("\n" + "=" * 60)
print("TEST 4: REAL-TIME FRAME PROCESSING (20ms frames)")
print("=" * 60)

frame_size = int(48000 * 0.02)  # 960 samples
test_frame = test_audio[:frame_size]

times = []
for _ in range(100):
    modified, proc_time = modifier.apply_modifications(
        test_frame,
        pitch_shift_semitones=2.0,
        energy_multiplier=1.2,
        pace_multiplier=1.0
    )
    times.append(proc_time)

times = np.array(times)
print(f"\n📊 Statistics (100 frames):")
print(f"  - Mean: {np.mean(times):.2f} ms")
print(f"  - Median: {np.median(times):.2f} ms")
print(f"  - 95th percentile: {np.percentile(times, 95):.2f} ms")

if np.mean(times) < 20:
    print("\n✓ Processing faster than real-time!")
else:
    print("\n⚠️  Processing slower than real-time")

print("\n" + "=" * 60)
print("NOTE: Listen to actual audio to evaluate quality")
print("This test only measures processing speed")
print("=" * 60)
