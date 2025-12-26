"""
Simple Passthrough Test

This is the simplest possible test - just pass microphone input directly to speakers.
If this works, the full system should work too.
"""

import sounddevice as sd
import numpy as np
import time

print("=" * 60)
print("SIMPLE AUDIO PASSTHROUGH TEST")
print("=" * 60)
print("\nThis will pass your microphone directly to speakers.")
print("You should hear yourself with a slight delay.")
print("\n⚠️  Use headphones to prevent feedback!")
print("\nRunning for 10 seconds...\n")

# Check permissions first
print("Checking audio devices...")
try:
    devices = sd.query_devices()
    print(f"✓ Found {len(devices)} audio devices")
    print(f"✓ Default input: {sd.default.device[0]}")
    print(f"✓ Default output: {sd.default.device[1]}")
except Exception as e:
    print(f"❌ Error accessing audio devices: {e}")
    print("\nThis usually means:")
    print("  1. No microphone/speakers connected")
    print("  2. Microphone permissions not granted")
    print("\nTo fix on macOS:")
    print("  System Preferences → Security & Privacy → Microphone")
    print("  Grant access to Terminal or Python")
    exit(1)

print("\nStarting passthrough...")
print("Speak into your microphone!\n")

frame_count = [0]  # Use list to modify in callback

def callback(indata, outdata, frames, time_info, status):
    if status:
        print(f"Status: {status}")
    
    # Simple passthrough
    outdata[:] = indata
    frame_count[0] += 1
    
    # Print progress every 50 frames (~1 second)
    if frame_count[0] % 50 == 0:
        print(f"  Processed {frame_count[0]} frames...")

try:
    with sd.Stream(
        samplerate=48000,
        channels=1,
        callback=callback,
        dtype=np.float32,
        blocksize=960  # 20ms frames
    ):
        print("✓ Stream started")
        print("Listening... (10 seconds)\n")
        time.sleep(10)
    
    print(f"\n✓ Test complete!")
    print(f"Total frames processed: {frame_count[0]}")
    
    if frame_count[0] == 0:
        print("\n❌ NO FRAMES PROCESSED!")
        print("\nThis means audio input is not working.")
        print("Possible issues:")
        print("  1. Microphone permissions not granted")
        print("  2. Wrong input device selected")
        print("  3. Microphone is muted")
        print("\nTo fix:")
        print("  - Check System Preferences → Security & Privacy → Microphone")
        print("  - Grant access to Terminal")
        print("  - Check System Preferences → Sound → Input")
        print("  - Ensure microphone volume is up")
    elif frame_count[0] < 400:  # Should be ~500 frames in 10 seconds
        print("\n⚠️  LOW FRAME COUNT")
        print("Expected ~500 frames, got", frame_count[0])
        print("Audio may be glitching or interrupted")
    else:
        print("\n✓ Frame count looks good!")
        heard = input("\nDid you hear yourself? (y/n): ").strip().lower()
        
        if heard == 'y':
            print("\n✓ Audio system is working!")
            print("The voice modifier should work too.")
            print("\nIf the voice modifier still doesn't work:")
            print("  1. Make sure you're speaking loud enough")
            print("  2. Wait a few seconds for processing to start")
            print("  3. Check that modifications are enabled")
        else:
            print("\n⚠️  Audio not heard")
            print("Possible issues:")
            print("  - Volume is too low")
            print("  - Wrong output device")
            print("  - Need to use headphones (feedback prevention)")

except KeyboardInterrupt:
    print("\n\nStopped by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nThis usually indicates:")
    print("  1. Microphone permissions not granted")
    print("  2. Audio device is in use by another app")
    print("  3. Incompatible audio settings")

print("\n" + "=" * 60)
