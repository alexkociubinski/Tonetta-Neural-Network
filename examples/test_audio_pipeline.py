"""
Test the audio pipeline independently.

This script tests:
- Microphone capture
- Frame-based processing
- Feature extraction speed
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from audio_pipeline import RealtimeAudioPipeline
import time

print("=" * 60)
print("AUDIO PIPELINE TEST")
print("=" * 60)
print("\nTesting microphone capture and feature extraction...")
print("Speak into your microphone!\n")

# Create pipeline
pipeline = RealtimeAudioPipeline(sample_rate=48000, frame_duration_ms=20)

# Start capturing
pipeline.start()

try:
    print(f"{'Time (s)':<10} {'RMS Energy':<15} {'Pitch (Hz)':<15} {'Speaking':<12} {'Proc Time (ms)':<15}")
    print("-" * 70)
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        features = pipeline.get_latest_features()
        
        if features:
            elapsed = time.time() - start_time
            pitch_str = f"{features['pitch_hz']:.1f}" if features['pitch_hz'] else "None"
            
            print(f"{elapsed:<10.2f} {features['rms_energy']:<15.4f} {pitch_str:<15} "
                  f"{features['speaking_rate']:<12.1f} {features['processing_time_ms']:<15.2f}")
            
            frame_count += 1
            
            if features['processing_time_ms'] > 15:
                print("  ⚠️  WARNING: Processing time > 15ms")
        
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n\nStopping...")
    pipeline.stop()
    
    elapsed = time.time() - start_time
    print(f"\n📊 Statistics:")
    print(f"  - Total frames: {frame_count}")
    print(f"  - Duration: {elapsed:.1f}s")
    print(f"  - Frame rate: {frame_count/elapsed:.1f} fps (target: 50 fps)")
    
    if frame_count/elapsed >= 45:
        print("  ✓ Performance good!")
    else:
        print("  ⚠️  Performance below target")
