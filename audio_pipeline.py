"""
STEP 1: REAL-TIME AUDIO PIPELINE

This module captures live microphone audio and extracts features in real-time.

PLAIN ENGLISH EXPLANATION:
- Think of this as a "listener" that constantly grabs tiny slices of your voice
- Each slice is 20 milliseconds (1/50th of a second)
- For each slice, we measure: How loud? How high-pitched? Are you speaking?
- These measurements become the "sensors" that will eventually control voice modifications
"""

import numpy as np
import sounddevice as sd
import librosa
from queue import Queue
from typing import Dict, Optional
import time


class AudioFeatureExtractor:
    """
    Extracts audio features from raw audio frames.
    
    WHAT THIS DOES:
    Converts raw sound waves into numbers that describe the sound:
    - RMS Energy: How loud the sound is (like a volume meter)
    - Pitch (F0): How high or low the voice is (measured in Hz)
    - Speaking Rate: Whether you're actively speaking (voice activity detection)
    """
    
    def __init__(self, sample_rate: int = 48000, frame_size: int = 960):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: How many audio samples per second (48000 = 48kHz, high quality)
            frame_size: How many samples per frame (960 samples = 20ms at 48kHz)
        
        PLAIN ENGLISH:
        - sample_rate is like video FPS, but for audio (48,000 "snapshots" per second)
        - frame_size determines how much audio we process at once (20ms chunks)
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_duration_ms = (frame_size / sample_rate) * 1000
        
        # For pitch detection, we need a slightly larger window
        self.pitch_buffer_size = 2048  # ~43ms at 48kHz
        self.pitch_buffer = np.zeros(self.pitch_buffer_size)
        
        print(f"✓ Feature extractor initialized:")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Frame size: {frame_size} samples ({self.frame_duration_ms:.1f} ms)")
    
    def extract_rms_energy(self, frame: np.ndarray) -> float:
        """
        Calculate RMS (Root Mean Square) energy - a measure of loudness.
        
        PLAIN ENGLISH:
        - Takes all the audio samples in the frame
        - Squares them (makes all values positive)
        - Takes the average
        - Takes the square root
        - Result: a single number representing "how loud"
        
        Returns:
            RMS energy value (0.0 = silence, higher = louder)
        """
        return np.sqrt(np.mean(frame ** 2))
    
    def extract_pitch(self, frame: np.ndarray, method: str = 'fast') -> Optional[float]:
        """
        Extract fundamental frequency (F0) - the pitch of the voice.
        
        METHODS:
        - 'fast': Autocorrelation-based (very fast, <1ms)
        - 'pyin': Probabilistic YIN (very accurate but slow, ~40ms)
        
        Returns:
            Pitch in Hz, or None if no pitch detected
        """
        # Update rolling buffer for pitch detection
        self.pitch_buffer = np.roll(self.pitch_buffer, -len(frame))
        self.pitch_buffer[-len(frame):] = frame
        
        if method == 'fast':
            # FAST AUTOCORRELATION METHOD
            try:
                # 1. Apply windowing
                windowed = self.pitch_buffer * np.hanning(len(self.pitch_buffer))
                
                # 2. Compute autocorrelation
                corr = np.correlate(windowed, windowed, mode='full')
                corr = corr[len(corr)//2:]
                
                # 3. Find peaks in a reasonable range (65Hz to 1000Hz)
                dmin = int(self.sample_rate / 1000)
                dmax = int(self.sample_rate / 65)
                
                if dmax > len(corr):
                    dmax = len(corr)
                
                # Only look in the range [dmin, dmax]
                search_region = corr[dmin:dmax]
                if len(search_region) == 0:
                    return None
                    
                peak = np.argmax(search_region) + dmin
                
                # 4. Check if peak is strong enough to be periodic
                if corr[peak] > 0.3 * corr[0]:  # Energy threshold
                    pitch = self.sample_rate / peak
                    return float(pitch)
            except:
                pass
                
        elif method == 'pyin':
            # HIGH-ACCURACY BUT SLOW METHOD
            try:
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    self.pitch_buffer,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=self.sample_rate,
                    frame_length=self.pitch_buffer_size
                )
                if f0 is not None and len(f0) > 0 and not np.isnan(f0[-1]):
                    return float(f0[-1])
            except:
                pass
        
        return None
    
    def extract_speaking_rate(self, frame: np.ndarray, energy_threshold: float = 0.01) -> float:
        """
        Detect voice activity - are you speaking right now?
        
        PLAIN ENGLISH:
        - Checks if the frame is loud enough to be speech
        - Returns 1.0 if speaking, 0.0 if silent
        - This is a simple proxy for "speaking rate" (more sophisticated version later)
        
        Args:
            frame: Audio frame to analyze
            energy_threshold: Minimum energy to consider as speech
        
        Returns:
            1.0 if speaking, 0.0 if silent
        """
        energy = self.extract_rms_energy(frame)
        return 1.0 if energy > energy_threshold else 0.0
    
    def extract_features(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Extract all features from a single audio frame.
        
        Returns:
            Dictionary with 'rms_energy', 'pitch_hz', 'speaking_rate'
        """
        start_time = time.time()
        
        features = {
            'rms_energy': self.extract_rms_energy(frame),
            'pitch_hz': self.extract_pitch(frame),
            'speaking_rate': self.extract_speaking_rate(frame)
        }
        
        # Track processing time to ensure we're fast enough
        processing_time_ms = (time.time() - start_time) * 1000
        features['processing_time_ms'] = processing_time_ms
        
        return features


class RealtimeAudioPipeline:
    """
    Captures live microphone audio and processes it in real-time.
    
    PLAIN ENGLISH:
    - This is the "main engine" that runs continuously
    - It grabs audio from your microphone in 20ms chunks
    - Sends each chunk to the feature extractor
    - Puts the results in a queue for the neural network to use later
    """
    
    def __init__(self, sample_rate: int = 48000, frame_duration_ms: int = 20):
        """
        Initialize the real-time audio pipeline.
        
        Args:
            sample_rate: Audio sample rate (48000 Hz = high quality)
            frame_duration_ms: How long each frame is (20ms = good balance)
        
        PLAIN ENGLISH:
        - 20ms frames mean we process audio 50 times per second
        - Smaller frames = lower latency but more CPU work
        - Larger frames = higher latency but easier processing
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Feature extractor
        self.feature_extractor = AudioFeatureExtractor(sample_rate, self.frame_size)
        
        # Queue for passing features to other components
        # Think of this as a conveyor belt between components
        self.feature_queue = Queue(maxsize=100)
        
        # Audio stream (will be initialized when we start)
        self.stream = None
        self.is_running = False
        
        print(f"\n✓ Real-time audio pipeline initialized")
        print(f"  - Processing {1000/frame_duration_ms:.0f} frames per second")
    
    def audio_callback(self, indata, frames, time_info, status):
        """
        This function is called automatically by sounddevice for each audio chunk.
        
        PLAIN ENGLISH:
        - sounddevice runs this function in the background constantly
        - 'indata' contains the raw audio samples from your microphone
        - We extract features and put them in the queue
        - This happens 50 times per second (every 20ms)
        
        IMPORTANT: This must be FAST (<20ms) or audio will glitch!
        """
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert to mono if stereo (take average of left/right channels)
        if indata.shape[1] > 1:
            audio_frame = np.mean(indata, axis=1)
        else:
            audio_frame = indata[:, 0]
        
        # Extract features from this frame
        features = self.feature_extractor.extract_features(audio_frame)
        
        # Put features in queue (non-blocking to avoid audio glitches)
        try:
            self.feature_queue.put_nowait(features)
        except:
            # Queue is full, skip this frame (better than blocking)
            pass
    
    def start(self):
        """
        Start capturing audio from the microphone.
        
        PLAIN ENGLISH:
        - Opens your microphone
        - Starts calling audio_callback() automatically every 20ms
        - Runs in the background until you call stop()
        """
        print("\n🎤 Starting microphone capture...")
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,  # Mono audio (simpler and faster)
            blocksize=self.frame_size,  # Process in 20ms chunks
            callback=self.audio_callback,
            dtype=np.float32
        )
        
        self.stream.start()
        self.is_running = True
        print("✓ Microphone active - speak to see features!\n")
    
    def stop(self):
        """Stop capturing audio."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
            print("\n✓ Microphone stopped")
    
    def get_latest_features(self) -> Optional[Dict[str, float]]:
        """
        Get the most recent features from the queue.
        
        Returns:
            Dictionary with features, or None if queue is empty
        """
        if not self.feature_queue.empty():
            return self.feature_queue.get()
        return None


# Demo: Run this file directly to test the audio pipeline
if __name__ == "__main__":
    print("=" * 60)
    print("TONETTA AUDIO PIPELINE - STEP 1 DEMO")
    print("=" * 60)
    print("\nThis will capture your microphone and show extracted features.")
    print("Speak into your microphone to see the values change!\n")
    
    # Create pipeline
    pipeline = RealtimeAudioPipeline(sample_rate=48000, frame_duration_ms=20)
    
    # Start capturing
    pipeline.start()
    
    try:
        print("Feature extraction running... (Ctrl+C to stop)\n")
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
                
                # Show warning if processing is too slow
                if features['processing_time_ms'] > 15:
                    print("  ⚠️  WARNING: Processing time > 15ms (may cause latency)")
            
            time.sleep(0.01)  # Small sleep to avoid busy-waiting
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
        pipeline.stop()
        
        # Show statistics
        elapsed = time.time() - start_time
        print(f"\n📊 Statistics:")
        print(f"  - Total frames processed: {frame_count}")
        print(f"  - Duration: {elapsed:.1f} seconds")
        print(f"  - Average frame rate: {frame_count/elapsed:.1f} fps")
        print(f"  - Target frame rate: 50 fps (20ms frames)")
        
        if frame_count/elapsed < 45:
            print("  ⚠️  Frame rate below target - system may be too slow")
        else:
            print("  ✓ Frame rate good - ready for real-time processing!")
