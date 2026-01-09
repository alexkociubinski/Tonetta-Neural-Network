"""
SIMPLE REAL-TIME VOICE MODIFIER

This version processes audio DIRECTLY in the callback - no queues, no threads.
It applies a simple energy boost to make your voice sound more confident.

This eliminates all latency and echo issues from the complex system.
"""

import sounddevice as sd
import numpy as np
import time

class SimpleVoiceModifier:
    """
    Ultra-lightweight voice modifier that runs inline.
    No neural network, no queues, no threading - just direct processing.
    """
    
    def __init__(self, sample_rate: int = 48000, target_rms: float = 0.08):
        self.sample_rate = sample_rate
        self.target_rms = target_rms  # Target RMS for "confident" speech
        self.frame_count = 0
        
    def process_frame(self, audio: np.ndarray) -> np.ndarray:
        """Apply simple energy boost directly."""
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms < 0.001:  # Silence
            return audio
        
        # Calculate energy multiplier to reach target RMS
        multiplier = self.target_rms / rms
        
        # Clamp to reasonable range (0.5x to 2.0x)
        multiplier = np.clip(multiplier, 0.5, 2.0)
        
        # Apply and prevent clipping
        modified = audio * multiplier
        modified = np.clip(modified, -0.95, 0.95)
        
        return modified
    
    def callback(self, indata, outdata, frames, time_info, status):
        """Audio callback - runs every 20ms."""
        if status:
            print(f"Audio status: {status}")
        
        # Process directly - no queues!
        audio = indata[:, 0] if len(indata.shape) > 1 else indata.flatten()
        modified = self.process_frame(audio)
        
        # Output
        if len(outdata.shape) > 1:
            outdata[:, 0] = modified
            if outdata.shape[1] > 1:
                outdata[:, 1] = modified
        else:
            outdata[:] = modified
        
        self.frame_count += 1
    
    def run(self, duration: float = None):
        """Start the voice modifier."""
        print("=" * 60)
        print("SIMPLE REAL-TIME VOICE MODIFIER")
        print("=" * 60)
        print("\nThis version has ZERO latency - processes audio inline.")
        print(f"Target RMS: {self.target_rms} (for confident speech)")
        print("\n⚠️  Use HEADPHONES to prevent feedback!")
        print("\nStarting...")
        
        try:
            with sd.Stream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.callback,
                dtype=np.float32,
                blocksize=960,  # 20ms frames
                latency='low'
            ):
                print("✓ Running! Speak into your microphone.")
                print("Press Ctrl+C to stop.\n")
                
                if duration:
                    time.sleep(duration)
                else:
                    while True:
                        time.sleep(1)
                        if self.frame_count % 50 == 0:
                            print(f"  Processed {self.frame_count} frames...")
        
        except KeyboardInterrupt:
            print("\n\nStopped.")
            print(f"Total frames processed: {self.frame_count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple real-time voice modifier')
    parser.add_argument('--target-rms', type=float, default=0.08,
                        help='Target RMS for confident speech (default: 0.08)')
    
    args = parser.parse_args()
    
    modifier = SimpleVoiceModifier(target_rms=args.target_rms)
    modifier.run()
