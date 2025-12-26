"""
Run the FAST/OPTIMIZED real-time voice modification system.

This version skips pitch detection for speed (<100ms latency).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from realtime_system_fast import FastRealtimeVoiceModifier
import time

print("=" * 60)
print("TONETTA OPTIMIZED VOICE MODIFIER")
print("=" * 60)
print("\nFast version - optimized for <100ms latency")
print("Trade-off: No pitch shifting (too slow)")
print("You'll hear energy/volume modifications only\n")

# Create system
modifier = FastRealtimeVoiceModifier(
    sample_rate=48000,
    frame_duration_ms=20,
    enable_pitch_shift=False  # Keep disabled for speed
)

# Start
modifier.start()

try:
    print("System running... (Ctrl+C to stop)\n")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    # Stop
    modifier.stop()
    
    # Show statistics
    modifier.print_statistics()
    
    print("\n" + "=" * 60)
    print("This is the FAST version for real-time use.")
    print("For full features (pitch/pace), use slower processing.")
    print("=" * 60)
