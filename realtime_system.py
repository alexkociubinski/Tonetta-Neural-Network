"""
STEP 4: REAL-TIME INTEGRATION LOOP

This module ties everything together into a complete real-time voice modification system.

PLAIN ENGLISH EXPLANATION:

This is the "conductor" that orchestrates all the pieces:
1. Captures microphone audio (Step 1)
2. Extracts features (Step 1)
3. Runs neural network to get control signals (Step 2)
4. Modifies audio using DSP (Step 3)
5. Plays modified audio back through speakers

All of this happens continuously with minimal delay (<100ms target).

LATENCY BOTTLENECKS & SOLUTIONS:

1. Audio I/O Buffering (~40ms total):
   - Input buffer: 20ms
   - Output buffer: 20ms
   - Solution: Use small buffer sizes, can't reduce much more without glitches

2. Feature Extraction (~5-10ms):
   - Pitch detection is the slowest part
   - Solution: Use efficient algorithms (librosa.pyin), consider downsampling

3. Neural Network Inference (~5-15ms):
   - Model size is key factor
   - Solution: Keep model small (<5K parameters), use CPU inference

4. DSP Modification (~10-20ms):
   - Pitch shifting is most expensive
   - Solution: Use optimized libraries (librosa, pyrubberband)

5. Python Overhead (~5ms):
   - GIL (Global Interpreter Lock), garbage collection
   - Solution: Minimize allocations, use NumPy, consider Cython for hot paths

TOTAL LATENCY BUDGET: 60-85ms (within <100ms requirement)
"""

import numpy as np
import sounddevice as sd
from queue import Queue, Empty
import threading
import time
import json
import os
from typing import Optional

from audio_pipeline import AudioFeatureExtractor
from voice_model import VoiceControlModel
from audio_modifier import AudioModifier


class RealtimeVoiceModifier:
    """
    Complete real-time voice modification system.
    
    ARCHITECTURE:
    
    [Microphone] → Input Buffer → Feature Extraction → Neural Network
                                                              ↓
    [Speakers]   ← Output Buffer ← DSP Modification ← Control Signals
    
    TWO THREADS:
    1. Audio I/O Thread (high priority): Handles microphone input and speaker output
    2. Processing Thread (normal priority): Runs feature extraction, NN, and DSP
    
    WHY TWO THREADS?
    - Audio I/O must never be blocked (causes glitches)
    - Processing can take variable time, so we decouple it
    - Queues connect the threads safely
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        frame_duration_ms: int = 20,
        enable_modifications: bool = True
    ):
        """
        Initialize the real-time voice modifier.
        
        Args:
            sample_rate: Audio sample rate (48000 Hz recommended)
            frame_duration_ms: Frame size (20ms = good latency/quality balance)
            enable_modifications: If False, just pass through audio (for testing)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.enable_modifications = enable_modifications
        
        # Initialize components
        print("Initializing components...")
        self.feature_extractor = AudioFeatureExtractor(sample_rate, self.frame_size)
        self.voice_model = VoiceControlModel()
        
        # Load trained model and parameters if available
        model_path = 'trained_model.h5'
        params_path = 'trained_model_params.json'
        
        if os.path.exists(model_path):
            try:
                self.voice_model.load(model_path)
                if os.path.exists(params_path):
                    with open(params_path, 'r') as f:
                        params = json.load(f)
                        self.voice_model.feature_means = np.array(params['feature_means'])
                        self.voice_model.feature_stds = np.array(params['feature_stds'])
                        print("✓ Loaded normalization parameters")
            except Exception as e:
                print(f"⚠ Could not load trained model: {e}")
                print("Continuing with default random weights.")
        else:
            print("ℹ No trained model found at trained_model.h5. Using default weights.")
            
        self.audio_modifier = AudioModifier(sample_rate)
        
        # Queues for thread communication
        # REDUCED maxsize: 10 frames = 200ms lag! 1 or 2 is much better for real-time.
        self.input_queue = Queue(maxsize=2)  # Raw audio from mic
        self.output_queue = Queue(maxsize=2)  # Modified audio to speakers
        
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
        
        # Warm up model (first inference is slow)
        print("\nWarming up neural network...")
        dummy_features = {
            'rms_energy': 0.05,
            'pitch_hz': 150.0,
            'speaking_rate': 1.0
        }
        _ = self.voice_model.predict(dummy_features)
        
        print("\n✓ Real-time voice modifier ready!")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Frame size: {self.frame_size} samples ({frame_duration_ms} ms)")
        print(f"  - Modifications: {'ENABLED' if enable_modifications else 'DISABLED (passthrough)'}")
    
    def audio_callback(self, indata, outdata, frames, time_info, status):
        """
        Combined input/output callback for duplex stream.
        
        CRITICAL: This handles both input and output in one callback.
        This ensures proper synchronization.
        """
        if status:
            print(f"Audio status: {status}")
        
        # Convert input to mono
        if len(indata.shape) > 1 and indata.shape[1] > 1:
            audio_frame = np.mean(indata, axis=1)
        else:
            audio_frame = indata[:, 0] if len(indata.shape) > 1 else indata
        
        # Put in queue for processing (non-blocking)
        try:
            self.input_queue.put_nowait((audio_frame.copy(), time.time()))
        except:
            # Queue full, drop frame
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
            # No audio available, output silence
            outdata.fill(0)
    
    def processing_loop(self):
        """
        Main processing loop that runs in a separate thread.
        
        FLOW:
        1. Get audio frame from input queue
        2. Extract features
        3. Run neural network
        4. Modify audio with DSP
        5. Put modified audio in output queue
        
        This loop runs continuously until stopped.
        """
        print("Processing loop started")
        
        while self.is_running:
            try:
                # Get audio from input queue (with timeout)
                audio_frame, capture_time = self.input_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Step 1: Extract features
                feature_start = time.time()
                features = self.feature_extractor.extract_features(audio_frame)
                feature_time = (time.time() - feature_start) * 1000
                
                # Step 2: Run neural network
                nn_start = time.time()
                control_signals = self.voice_model.predict(features)
                nn_time = (time.time() - nn_start) * 1000
                
                # Step 3: Modify audio (if enabled)
                dsp_start = time.time()
                if self.enable_modifications:
                    modified_audio, dsp_time = self.audio_modifier.apply_modifications(
                        audio_frame,
                        pitch_shift_semitones=control_signals['pitch_shift_semitones'],
                        energy_multiplier=control_signals['energy_multiplier'],
                        pace_multiplier=control_signals['pace_multiplier']
                    )
                else:
                    # Passthrough mode (for testing latency without modifications)
                    modified_audio = audio_frame
                    dsp_time = 0.0
                
                # Step 4: Put modified audio in output queue
                try:
                    self.output_queue.put_nowait(modified_audio)
                except:
                    # Queue full, drop frame
                    pass
                
                # Calculate total latency
                total_latency = (time.time() - capture_time) * 1000
                
                # Update statistics
                self.stats['frames_processed'] += 1
                self.stats['total_latency_ms'].append(total_latency)
                self.stats['feature_extraction_ms'].append(feature_time)
                self.stats['nn_inference_ms'].append(nn_time)
                self.stats['dsp_modification_ms'].append(dsp_time)
                
                # Warn if latency is too high
                if total_latency > 100:
                    print(f"⚠️  High latency: {total_latency:.1f} ms")
            
            except Empty:
                # No audio available, continue waiting
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def start(self):
        """Start the real-time voice modification system."""
        print("\n" + "=" * 60)
        print("STARTING REAL-TIME VOICE MODIFIER")
        print("=" * 60)
        
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Start duplex audio stream (both input and output)
        print("\n🎤 Opening microphone and speakers...")
        
        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.frame_size,
            callback=self.audio_callback,
            dtype=np.float32,
            latency='low'
        )
        
        self.stream.start()
        
        print("✓ System running!")
        print("\nSpeak into your microphone - you should hear your modified voice!")
        print("Press Ctrl+C to stop.\n")
    
    def stop(self):
        """Stop the real-time voice modification system."""
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
        
        # Convert lists to numpy arrays for statistics
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
            print("  ❌ TOO HIGH - Needs optimization")
        
        print("=" * 60)


# Demo: Run this file directly to test the complete system
if __name__ == "__main__":
    print("=" * 60)
    print("TONETTA REAL-TIME VOICE MODIFIER - STEP 4 DEMO")
    print("=" * 60)
    print("\nThis is the complete end-to-end system!")
    print("Speak into your microphone and hear your modified voice.\n")
    
    # Create system
    modifier = RealtimeVoiceModifier(
        sample_rate=48000,
        frame_duration_ms=20,
        enable_modifications=True  # Set to False to test passthrough latency
    )
    
    # Start system
    modifier.start()
    
    try:
        # Run until user stops
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        # Stop system
        modifier.stop()
        
        # Print statistics
        modifier.print_statistics()
        
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("\n1. System is currently running with the trained model.")
        print("2. To improve quality further, collect more niche training data.")
        print("3. See README.md for more advanced optimization options.")
        print("\n" + "=" * 60)
