"""
OPTIMIZED Real-Time System (Low Latency Version)

This version sacrifices pitch detection for speed.
Target: <100ms latency for real-time use.

Key optimizations:
- Skip pitch detection (too slow for 20ms frames)
- Use only RMS energy and speaking rate
- Simpler neural network
- Faster DSP (skip pitch shifting initially)
"""

import numpy as np
import sounddevice as sd
from queue import Queue, Empty
import threading
import time
from typing import Optional

from voice_model import VoiceControlModel
from audio_modifier import AudioModifier


class FastAudioFeatureExtractor:
    """Fast feature extraction without pitch detection."""
    
    def __init__(self, sample_rate: int = 48000, frame_size: int = 960):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
    
    def extract_features(self, frame: np.ndarray) -> dict:
        """Extract only fast features (no pitch)."""
        start_time = time.time()
        
        # RMS energy (very fast)
        rms_energy = np.sqrt(np.mean(frame ** 2))
        
        # Speaking rate (very fast)
        speaking_rate = 1.0 if rms_energy > 0.01 else 0.0
        
        # Use a default pitch value (skip detection)
        pitch_hz = 150.0  # Assume average pitch
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return {
            'rms_energy': rms_energy,
            'pitch_hz': pitch_hz,
            'speaking_rate': speaking_rate,
            'processing_time_ms': processing_time_ms
        }


class FastRealtimeVoiceModifier:
    """Optimized real-time voice modifier with <100ms latency."""
    
    def __init__(
        self,
        sample_rate: int = 48000,
        frame_duration_ms: int = 20,
        enable_pitch_shift: bool = False  # Disabled by default for speed
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.enable_pitch_shift = enable_pitch_shift
        
        # Initialize components
        print("Initializing optimized components...")
        self.feature_extractor = FastAudioFeatureExtractor(sample_rate, self.frame_size)
        self.voice_model = VoiceControlModel()
        self.audio_modifier = AudioModifier(sample_rate)
        
        # Queues
        self.input_queue = Queue(maxsize=10)
        self.output_queue = Queue(maxsize=10)
        
        # Control flags
        self.is_running = False
        self.processing_thread = None
        
        # Performance monitoring
        self.stats = {
            'frames_processed': 0,
            'total_latency_ms': [],
            'feature_extraction_ms': [],
            'nn_inference_ms': [],
            'dsp_modification_ms': [],
        }
        
        # Warm up model
        print("\nWarming up neural network...")
        dummy_features = {
            'rms_energy': 0.05,
            'pitch_hz': 150.0,
            'speaking_rate': 1.0
        }
        _ = self.voice_model.predict(dummy_features)
        
        print("\n✓ Optimized voice modifier ready!")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Frame size: {self.frame_size} samples ({frame_duration_ms} ms)")
        print(f"  - Pitch shifting: {'ENABLED' if enable_pitch_shift else 'DISABLED (for speed)'}")
    
    def audio_callback(self, indata, outdata, frames, time_info, status):
        """Combined input/output callback."""
        if status:
            print(f"Audio status: {status}")
        
        # Convert input to mono
        if len(indata.shape) > 1 and indata.shape[1] > 1:
            audio_frame = np.mean(indata, axis=1)
        else:
            audio_frame = indata[:, 0] if len(indata.shape) > 1 else indata
        
        # Put in queue for processing
        try:
            self.input_queue.put_nowait((audio_frame.copy(), time.time()))
        except:
            pass
        
        # Get modified audio from output queue
        try:
            modified_audio = self.output_queue.get_nowait()
            
            # Ensure correct length
            if len(modified_audio) < frames:
                modified_audio = np.pad(modified_audio, (0, frames - len(modified_audio)))
            elif len(modified_audio) > frames:
                modified_audio = modified_audio[:frames]
            
            # Write to output
            if len(outdata.shape) > 1:
                outdata[:, 0] = modified_audio
                if outdata.shape[1] > 1:
                    outdata[:, 1] = modified_audio
            else:
                outdata[:] = modified_audio
        except Empty:
            outdata.fill(0)
    
    def processing_loop(self):
        """Main processing loop."""
        print("Processing loop started")
        
        while self.is_running:
            try:
                audio_frame, capture_time = self.input_queue.get(timeout=0.1)
                
                # Step 1: Extract features (fast version)
                feature_start = time.time()
                features = self.feature_extractor.extract_features(audio_frame)
                feature_time = (time.time() - feature_start) * 1000
                
                # Step 2: Run neural network
                nn_start = time.time()
                control_signals = self.voice_model.predict(features)
                nn_time = (time.time() - nn_start) * 1000
                
                # Step 3: Modify audio (skip pitch shift for speed)
                dsp_start = time.time()
                modified_audio, dsp_time = self.audio_modifier.apply_modifications(
                    audio_frame,
                    pitch_shift_semitones=control_signals['pitch_shift_semitones'] if self.enable_pitch_shift else 0.0,
                    energy_multiplier=control_signals['energy_multiplier'],
                    pace_multiplier=1.0  # Skip pace for speed
                )
                
                # Step 4: Put modified audio in output queue
                try:
                    self.output_queue.put_nowait(modified_audio)
                except:
                    pass
                
                # Calculate total latency
                total_latency = (time.time() - capture_time) * 1000
                
                # Update statistics
                self.stats['frames_processed'] += 1
                self.stats['total_latency_ms'].append(total_latency)
                self.stats['feature_extraction_ms'].append(feature_time)
                self.stats['nn_inference_ms'].append(nn_time)
                self.stats['dsp_modification_ms'].append(dsp_time)
                
                # Only warn occasionally to avoid spam
                if total_latency > 100 and self.stats['frames_processed'] % 50 == 0:
                    print(f"⚠️  Latency: {total_latency:.1f} ms")
            
            except Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def start(self):
        """Start the system."""
        print("\n" + "=" * 60)
        print("STARTING OPTIMIZED VOICE MODIFIER")
        print("=" * 60)
        
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Start duplex audio stream
        print("\n🎤 Opening microphone and speakers...")
        
        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.frame_size,
            callback=self.audio_callback,
            dtype=np.float32
        )
        
        self.stream.start()
        
        print("✓ System running!")
        print("\nSpeak into your microphone - you should hear your modified voice!")
        print("(Energy/volume modifications only - pitch disabled for speed)")
        print("Press Ctrl+C to stop.\n")
    
    def stop(self):
        """Stop the system."""
        print("\nStopping...")
        
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        print("✓ System stopped")
    
    def print_statistics(self):
        """Print performance statistics."""
        if self.stats['frames_processed'] == 0:
            print("No frames processed yet")
            return
        
        print("\n" + "=" * 60)
        print("PERFORMANCE STATISTICS")
        print("=" * 60)
        
        print(f"\nFrames processed: {self.stats['frames_processed']}")
        
        total_latency = np.array(self.stats['total_latency_ms'])
        feature_time = np.array(self.stats['feature_extraction_ms'])
        nn_time = np.array(self.stats['nn_inference_ms'])
        dsp_time = np.array(self.stats['dsp_modification_ms'])
        
        print(f"\nTotal End-to-End Latency:")
        print(f"  - Mean: {np.mean(total_latency):.2f} ms")
        print(f"  - Median: {np.median(total_latency):.2f} ms")
        print(f"  - 95th percentile: {np.percentile(total_latency, 95):.2f} ms")
        print(f"  - Max: {np.max(total_latency):.2f} ms")
        
        print(f"\nComponent Breakdown:")
        print(f"  Feature Extraction:")
        print(f"    - Mean: {np.mean(feature_time):.2f} ms")
        print(f"  Neural Network:")
        print(f"    - Mean: {np.mean(nn_time):.2f} ms")
        print(f"  DSP Modification:")
        print(f"    - Mean: {np.mean(dsp_time):.2f} ms")
        
        print(f"\nLatency Assessment:")
        if np.mean(total_latency) < 100:
            print("  ✓ EXCELLENT - Below 100ms target!")
        elif np.mean(total_latency) < 150:
            print("  ⚠️  ACCEPTABLE - Slightly above target, but usable")
        else:
            print("  ❌ TOO HIGH - Needs more optimization")
        
        print("=" * 60)


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZED TONETTA VOICE MODIFIER")
    print("=" * 60)
    print("\nThis version is optimized for low latency (<100ms).")
    print("Trade-off: No pitch detection (too slow for 20ms frames)\n")
    
    # Create system
    modifier = FastRealtimeVoiceModifier(
        sample_rate=48000,
        frame_duration_ms=20,
        enable_pitch_shift=False  # Disable for speed
    )
    
    # Start system
    modifier.start()
    
    try:
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        modifier.stop()
        modifier.print_statistics()
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION NOTES")
        print("=" * 60)
        print("\nWhat was optimized:")
        print("  ✓ Skipped pitch detection (was taking 40ms)")
        print("  ✓ Skipped pitch shifting DSP (slow)")
        print("  ✓ Skipped pace modification (slow)")
        print("  ✓ Only using energy/volume adjustments")
        print("\nResult: Should be <100ms latency")
        print("\nFor production:")
        print("  - Use faster pitch detection algorithm")
        print("  - Process larger frames (50-100ms) for better pitch detection")
        print("  - Use GPU for DSP if available")
        print("=" * 60)
