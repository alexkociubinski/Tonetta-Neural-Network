"""
DATA COLLECTION TOOL

Record practice calls and label them for training.

USAGE:
1. Run this script
2. Speak into microphone for 30 seconds
3. Rate your confidence (1-10)
4. Recording is saved with label
5. Repeat to build dataset
"""

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import json
import os
from datetime import datetime


class DataRecorder:
    """Records audio and collects labels for training."""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.recordings_dir = 'training_data/recordings'
        self.labels_file = 'training_data/labels.json'
        
        # Create directories
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Load existing labels
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r') as f:
                self.labels = json.load(f)
        else:
            self.labels = []
    
    def record_audio(self, duration: int = 30) -> np.ndarray:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
        
        Returns:
            Audio samples
        """
        print(f"\n🎤 Recording for {duration} seconds...")
        print("Speak naturally, as if on a call.")
        print("Recording starts in 3... 2... 1...\n")
        
        # Record
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        
        print("✓ Recording complete!")
        return audio[:, 0]
    
    def collect_labels(self) -> dict:
        """
        Collect labels from user.
        
        Returns:
            Dictionary with labels
        """
        print("\n" + "=" * 60)
        print("LABEL THIS RECORDING")
        print("=" * 60)
        
        # Confidence rating
        while True:
            try:
                confidence = int(input("\nConfidence level (1-10, 10=very confident): "))
                if 1 <= confidence <= 10:
                    break
                print("Please enter a number between 1 and 10")
            except ValueError:
                print("Please enter a valid number")
        
        # Optional: Energy level
        while True:
            try:
                energy = int(input("Energy level (1-10, 10=very energetic): "))
                if 1 <= energy <= 10:
                    break
                print("Please enter a number between 1 and 10")
            except ValueError:
                print("Please enter a valid number")
        
        # Optional: Pace
        while True:
            try:
                pace = int(input("Speaking pace (1-10, 5=normal, 10=very fast): "))
                if 1 <= pace <= 10:
                    break
                print("Please enter a number between 1 and 10")
            except ValueError:
                print("Please enter a valid number")
        
        # Notes
        notes = input("\nOptional notes about this recording: ").strip()
        
        return {
            'confidence': confidence / 10.0,  # Normalize to 0-1
            'energy': energy / 10.0,
            'pace': pace / 10.0,
            'notes': notes
        }
    
    def save_recording(self, audio: np.ndarray, labels: dict):
        """Save recording and labels."""
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        filepath = os.path.join(self.recordings_dir, filename)
        
        # Save audio
        wav.write(filepath, self.sample_rate, audio)
        
        # Save labels
        label_entry = {
            'filename': filename,
            'filepath': filepath,
            'timestamp': timestamp,
            **labels
        }
        self.labels.append(label_entry)
        
        with open(self.labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)
        
        print(f"\n✓ Saved: {filename}")
        print(f"✓ Total recordings: {len(self.labels)}")
    
    def run_collection_session(self, num_recordings: int = 5):
        """Run a data collection session."""
        print("=" * 60)
        print("DATA COLLECTION SESSION")
        print("=" * 60)
        print(f"\nWe'll record {num_recordings} samples.")
        print("Each recording is 30 seconds.")
        print("After each recording, you'll rate your confidence/energy/pace.\n")
        
        input("Press Enter when ready to start...")
        
        for i in range(num_recordings):
            print(f"\n{'=' * 60}")
            print(f"RECORDING {i+1} of {num_recordings}")
            print("=" * 60)
            
            # Record
            audio = self.record_audio(duration=30)
            
            # Collect labels
            labels = self.collect_labels()
            
            # Save
            self.save_recording(audio, labels)
            
            if i < num_recordings - 1:
                print("\nTake a short break before the next recording...")
                input("Press Enter to continue...")
        
        print("\n" + "=" * 60)
        print("SESSION COMPLETE!")
        print("=" * 60)
        print(f"\nTotal recordings collected: {len(self.labels)}")
        print(f"Saved to: {self.recordings_dir}")
        print(f"Labels saved to: {self.labels_file}")
        print("\nNext step: Run train_model.py to train on this data")


if __name__ == "__main__":
    recorder = DataRecorder()
    
    print("=" * 60)
    print("TONETTA DATA COLLECTION TOOL")
    print("=" * 60)
    print("\nThis tool helps you build a training dataset.")
    print("You'll record yourself speaking and label each recording.\n")
    
    # Ask how many recordings
    while True:
        try:
            num = int(input("How many recordings do you want to make? (recommended: 10-20): "))
            if num > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Run session
    recorder.run_collection_session(num_recordings=num)
