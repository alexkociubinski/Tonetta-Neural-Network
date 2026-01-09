"""
STEP 3: AUDIO MODIFICATION ENGINE

This module applies DSP (Digital Signal Processing) to modify audio in real-time.

PLAIN ENGLISH EXPLANATION:

The neural network from Step 2 doesn't directly create audio - it just decides
"how much" to modify the voice. This module does the actual audio modification.

WHY SEPARATE DSP FROM NEURAL NETWORK?
1. SPEED: DSP algorithms are highly optimized C code (librosa uses Cython)
2. QUALITY: Phase vocoders produce better audio than neural synthesis
3. CONTROL: Neural network is the "brain", DSP is the "hands"
4. MODULARITY: Can swap DSP algorithms without retraining the network

WHAT WE'RE MODIFYING:
- Pitch: How high or low the voice sounds
- Energy: How loud the voice is
- Pace: How fast you're speaking
"""

import numpy as np
import librosa
import pyrubberband as pyrb
from typing import Optional
import time


class AudioModifier:
    """
    Applies real-time audio modifications using DSP techniques.
    
    THREE CORE OPERATIONS:
    1. Pitch shifting - change voice pitch without changing speed
    2. Energy adjustment - make voice louder or quieter
    3. Time-scale modification - change speaking speed without changing pitch
    """
    
    def __init__(self, sample_rate: int = 48000):
        """
        Initialize the audio modifier.
        
        Args:
            sample_rate: Audio sample rate (must match pipeline)
        """
        self.sample_rate = sample_rate
        
        print("✓ Audio modifier initialized")
        print(f"  - Sample rate: {sample_rate} Hz")
    
    def shift_pitch(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """
        Shift the pitch of audio without changing duration.
        
        PLAIN ENGLISH:
        - Makes your voice higher or lower
        - Positive semitones = higher pitch (chipmunk effect at +12)
        - Negative semitones = lower pitch (Darth Vader effect at -12)
        - 1 semitone = one piano key (12 semitones = one octave)
        
        HOW IT WORKS (Phase Vocoder):
        1. Convert audio to frequency domain (FFT - Fast Fourier Transform)
        2. Shift all frequencies up or down
        3. Convert back to time domain (inverse FFT)
        4. Preserve timing by resampling
        
        Args:
            audio: Input audio samples
            semitones: Pitch shift in semitones (±12 is typical range)
        
        Returns:
            Pitch-shifted audio (same length as input)
        """
        if abs(semitones) < 0.1:  # Skip if shift is negligible
            return audio
        
        try:
            # Use librosa's pitch shift with optimized real-time parameters
            shifted = librosa.effects.pitch_shift(
                y=audio,
                sr=self.sample_rate,
                n_steps=semitones,
                bins_per_octave=12,
                n_fft=512,      # Smaller FFT for 960-sample frames
                hop_length=128  # Smaller hop for better real-time phase vocoding
            )
            return shifted
        except Exception as e:
            print(f"Pitch shift error: {e}")
            return audio  # Return original on error
    
    def adjust_energy(self, audio: np.ndarray, multiplier: float) -> np.ndarray:
        """
        Adjust the energy (volume) of audio.
        
        PLAIN ENGLISH:
        - This is the simplest operation: just multiply all samples
        - multiplier = 2.0 → twice as loud (careful, can clip!)
        - multiplier = 0.5 → half as loud
        - multiplier = 1.0 → no change
        
        IMPORTANT: We clip to [-1, 1] to prevent distortion
        
        Args:
            audio: Input audio samples
            multiplier: Energy multiplier (0.5 to 2.0 typical range)
        
        Returns:
            Volume-adjusted audio
        """
        if abs(multiplier - 1.0) < 0.01:  # Skip if negligible change
            return audio
        
        # Simple multiplication
        adjusted = audio * multiplier
        
        # Clip to prevent distortion (audio must stay in [-1, 1] range)
        adjusted = np.clip(adjusted, -1.0, 1.0)
        
        return adjusted
    
    def adjust_pace(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """
        Adjust speaking pace (time-scale modification).
        
        PLAIN ENGLISH:
        - Makes you speak faster or slower WITHOUT changing pitch
        - rate = 1.2 → 20% faster (like fast-forwarding)
        - rate = 0.8 → 20% slower (like slow-motion)
        - rate = 1.0 → no change
        
        HOW IT WORKS (Rubber Band / Phase Vocoder):
        1. Identify "stable" regions (voiced speech)
        2. Stretch or compress time by duplicating/removing samples intelligently
        3. Preserve pitch by adjusting phase relationships
        
        NOTE: This changes the output length!
        - rate > 1.0 → shorter output (faster speech)
        - rate < 1.0 → longer output (slower speech)
        
        Args:
            audio: Input audio samples
            rate: Time-stretch rate (0.8 to 1.2 typical range)
        
        Returns:
            Time-stretched audio (length = original_length / rate)
        """
        if abs(rate - 1.0) < 0.01:  # Skip if negligible change
            return audio
        
        try:
            # Use pyrubberband for high-quality time stretching
            # This is the same library used by professional audio software
            stretched = pyrb.time_stretch(audio, self.sample_rate, rate)
            return stretched
        except Exception as e:
            # Fallback to librosa if pyrubberband fails
            try:
                stretched = librosa.effects.time_stretch(y=audio, rate=rate)
                return stretched
            except:
                print(f"Pace adjustment error: {e}")
                return audio  # Return original on error
    
    def apply_modifications(
        self,
        audio: np.ndarray,
        pitch_shift_semitones: float = 0.0,
        energy_multiplier: float = 1.0,
        pace_multiplier: float = 1.0
    ) -> np.ndarray:
        """
        Apply all modifications to an audio frame.
        
        ORDER MATTERS:
        1. Pitch shift first (preserves timing)
        2. Pace adjustment second (changes timing)
        3. Energy adjustment last (simple multiplication)
        
        Args:
            audio: Input audio frame
            pitch_shift_semitones: Pitch shift from neural network
            energy_multiplier: Energy boost from neural network
            pace_multiplier: Pace adjustment from neural network
        
        Returns:
            Modified audio (may be different length due to pace adjustment)
        """
        start_time = time.time()
        
        # Step 1: Pitch shift
        modified = self.shift_pitch(audio, pitch_shift_semitones)
        
        # Step 2: Pace adjustment
        modified = self.adjust_pace(modified, pace_multiplier)
        
        # Step 3: Energy adjustment
        modified = self.adjust_energy(modified, energy_multiplier)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return modified, processing_time_ms
    
    def apply_modifications_streaming(
        self,
        audio: np.ndarray,
        control_signals: dict
    ) -> np.ndarray:
        """
        Apply modifications optimized for streaming (frame-by-frame).
        
        PLAIN ENGLISH:
        - Takes control signals directly from neural network
        - Applies modifications in real-time
        - Optimized for low latency
        
        Args:
            audio: Input audio frame
            control_signals: Dictionary from voice_model.py
        
        Returns:
            Modified audio frame
        """
        return self.apply_modifications(
            audio,
            pitch_shift_semitones=control_signals['pitch_shift_semitones'],
            energy_multiplier=control_signals['energy_multiplier'],
            pace_multiplier=control_signals['pace_multiplier']
        )


# Demo: Run this file directly to test audio modifications
if __name__ == "__main__":
    print("=" * 60)
    print("TONETTA AUDIO MODIFIER - STEP 3 DEMO")
    print("=" * 60)
    print("\nThis demonstrates DSP-based audio modifications.\n")
    
    # Create modifier
    modifier = AudioModifier(sample_rate=48000)
    
    # Generate test audio (1 second of 440 Hz sine wave - musical note A)
    print("Generating test audio (440 Hz sine wave)...")
    duration = 1.0
    t = np.linspace(0, duration, int(48000 * duration))
    test_audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
    
    print("\n" + "=" * 60)
    print("TEST 1: PITCH SHIFTING")
    print("=" * 60)
    
    pitch_shifts = [-12, -5, 0, +5, +12]  # Semitones
    
    for semitones in pitch_shifts:
        start = time.time()
        shifted = modifier.shift_pitch(test_audio, semitones)
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"  Shift {semitones:+3d} semitones: {elapsed_ms:6.2f} ms")
    
    print("\n" + "=" * 60)
    print("TEST 2: ENERGY ADJUSTMENT")
    print("=" * 60)
    
    energy_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0]
    
    for mult in energy_multipliers:
        start = time.time()
        adjusted = modifier.adjust_energy(test_audio, mult)
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"  Multiplier {mult:.2f}x: {elapsed_ms:6.2f} ms")
    
    print("\n" + "=" * 60)
    print("TEST 3: PACE ADJUSTMENT")
    print("=" * 60)
    
    pace_rates = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for rate in pace_rates:
        start = time.time()
        stretched = modifier.adjust_pace(test_audio, rate)
        elapsed_ms = (time.time() - start) * 1000
        
        output_duration = len(stretched) / 48000
        print(f"  Rate {rate:.1f}x: {elapsed_ms:6.2f} ms (output: {output_duration:.2f}s)")
    
    print("\n" + "=" * 60)
    print("TEST 4: COMBINED MODIFICATIONS")
    print("=" * 60)
    
    # Test realistic modification combinations
    test_cases = [
        {"pitch": +2, "energy": 1.2, "pace": 1.0, "name": "Confident boost"},
        {"pitch": -1, "energy": 0.9, "pace": 0.95, "name": "Calm down"},
        {"pitch": +3, "energy": 1.3, "pace": 1.1, "name": "Energetic"},
    ]
    
    for case in test_cases:
        modified, proc_time = modifier.apply_modifications(
            test_audio,
            pitch_shift_semitones=case["pitch"],
            energy_multiplier=case["energy"],
            pace_multiplier=case["pace"]
        )
        
        print(f"  {case['name']:20s}: {proc_time:6.2f} ms")
    
    print("\n" + "=" * 60)
    print("LATENCY ANALYSIS")
    print("=" * 60)
    
    # Test on realistic frame size (20ms)
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
    print(f"\n20ms frame processing (100 iterations):")
    print(f"  - Mean: {np.mean(times):.2f} ms")
    print(f"  - Median: {np.median(times):.2f} ms")
    print(f"  - 95th percentile: {np.percentile(times, 95):.2f} ms")
    
    if np.mean(times) < 20:
        print("\n✓ Processing faster than real-time (good!)")
    else:
        print("\n⚠️  Processing slower than real-time (may cause latency)")
    
    print("\n" + "=" * 60)
    print("NOTE: Actual audio quality should be evaluated by listening.")
    print("This demo only measures processing speed.")
    print("=" * 60)
