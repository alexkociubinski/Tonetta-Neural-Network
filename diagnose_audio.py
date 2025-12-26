"""
Audio System Diagnostic Tool

This script helps diagnose audio input/output issues.
"""

import sounddevice as sd
import numpy as np

print("=" * 60)
print("AUDIO SYSTEM DIAGNOSTICS")
print("=" * 60)

# List all audio devices
print("\n📋 Available Audio Devices:")
print("-" * 60)
devices = sd.query_devices()
print(devices)

# Get default devices
print("\n🎯 Default Devices:")
print("-" * 60)
print(f"Input (microphone): {sd.default.device[0]}")
print(f"Output (speakers): {sd.default.device[1]}")

# Test microphone
print("\n🎤 Testing Microphone (5 seconds)...")
print("Speak into your microphone now!")
print("-" * 60)

duration = 5
sample_rate = 48000

try:
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32
    )
    sd.wait()
    
    # Analyze recording
    audio = recording[:, 0]
    rms = np.sqrt(np.mean(audio ** 2))
    max_amplitude = np.max(np.abs(audio))
    
    print(f"\n✓ Recording complete!")
    print(f"  - RMS Energy: {rms:.6f}")
    print(f"  - Max Amplitude: {max_amplitude:.6f}")
    
    if rms < 0.001:
        print("\n⚠️  WARNING: Very low audio level detected!")
        print("Possible issues:")
        print("  1. Microphone is muted or volume is very low")
        print("  2. Wrong input device selected")
        print("  3. Microphone permissions not granted")
        print("\nTo fix:")
        print("  - Check System Preferences → Sound → Input")
        print("  - Ensure microphone volume is up")
        print("  - Grant microphone permissions to Terminal/Python")
    elif rms < 0.01:
        print("\n⚠️  Audio level is low but detected")
        print("  - Try speaking louder or moving closer to mic")
    else:
        print("\n✓ Audio level looks good!")
    
except Exception as e:
    print(f"\n❌ Error recording audio: {e}")
    print("\nPossible issues:")
    print("  1. No microphone connected")
    print("  2. Microphone permissions not granted")
    print("  3. Audio device is in use by another application")

# Test playback
print("\n🔊 Testing Speakers (playing test tone)...")
print("-" * 60)

try:
    # Generate 440 Hz sine wave (musical note A)
    t = np.linspace(0, 2, int(sample_rate * 2))
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    print("Playing 440 Hz tone for 2 seconds...")
    sd.play(tone, sample_rate)
    sd.wait()
    
    print("✓ Playback complete!")
    heard = input("\nDid you hear the tone? (y/n): ").strip().lower()
    
    if heard != 'y':
        print("\n⚠️  Speaker issue detected!")
        print("Possible issues:")
        print("  1. Volume is muted or very low")
        print("  2. Wrong output device selected")
        print("  3. Headphones not connected")
        print("\nTo fix:")
        print("  - Check System Preferences → Sound → Output")
        print("  - Ensure volume is up")
        print("  - Check correct output device is selected")
    else:
        print("\n✓ Speakers working!")
        
except Exception as e:
    print(f"\n❌ Error playing audio: {e}")

# Test simultaneous input/output (like real-time system)
print("\n🔄 Testing Simultaneous Input/Output (5 seconds)...")
print("Speak into your microphone - you should hear yourself!")
print("-" * 60)

try:
    # Simple passthrough test
    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"Status: {status}")
        outdata[:] = indata
    
    with sd.Stream(
        samplerate=sample_rate,
        channels=1,
        callback=callback,
        dtype=np.float32
    ):
        print("Passthrough active... speak now!")
        sd.sleep(5000)
    
    print("\n✓ Simultaneous I/O test complete!")
    heard_self = input("Did you hear yourself? (y/n): ").strip().lower()
    
    if heard_self != 'y':
        print("\n⚠️  Real-time audio issue detected!")
        print("This is the same issue affecting the voice modifier.")
        print("\nPossible causes:")
        print("  1. Audio feedback prevention (system is blocking loopback)")
        print("  2. Latency is too high (you spoke but didn't wait long enough)")
        print("  3. Input/output device mismatch")
        print("\nRecommendations:")
        print("  - Use headphones (prevents feedback)")
        print("  - Check both input and output devices are correct")
        print("  - Try increasing buffer size for stability")
    else:
        print("\n✓ Real-time audio working!")
        print("The voice modifier should work. If it doesn't:")
        print("  1. Check that modifications are enabled")
        print("  2. Ensure you're speaking loud enough")
        print("  3. Wait a moment for processing to start")
        
except Exception as e:
    print(f"\n❌ Error in simultaneous I/O: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
print("\nIf you're still having issues:")
print("  1. Check macOS System Preferences → Security & Privacy → Microphone")
print("  2. Ensure Terminal/Python has microphone access")
print("  3. Try restarting the application")
print("  4. Check that no other app is using the microphone")
print("=" * 60)
