"""
Run the complete real-time voice modification system.

This is the main demo that shows everything working together.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from realtime_system import RealtimeVoiceModifier
import time

print("=" * 60)
print("TONETTA REAL-TIME VOICE MODIFIER")
print("=" * 60)
print("\nComplete end-to-end system demonstration")
print("Speak into your microphone to hear your modified voice!\n")

# Create system
modifier = RealtimeVoiceModifier(
    sample_rate=48000,
    frame_duration_ms=20,
    enable_modifications=True  # Set to False to test passthrough
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
    print("Thank you for testing Tonetta!")
    print("=" * 60)
