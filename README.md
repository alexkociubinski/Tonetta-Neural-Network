# Tonetta - Real-Time Voice Modification System

**A production-ready foundation for real-time voice coaching during calls**

Tonetta adjusts your voice in real-time during calls (confidence, pitch, energy, pace) with extremely low latency (<100ms). This is the foundational system designed for a startup, not a tutorial toy.

---

## 🎯 What This System Does

**Current Capabilities:**
- ✅ Captures live microphone audio in 20ms frames
- ✅ Extracts audio features (RMS energy, pitch, speaking rate)
- ✅ Runs lightweight neural network for control signal generation
- ✅ Modifies audio using DSP (pitch shift, energy boost, pace adjustment)
- ✅ Plays back modified audio with <100ms latency

**What It Doesn't Do Yet:**
- ❌ Train the neural network (weights are currently random)
- ❌ Detect confidence levels
- ❌ Match pace to conversation partner
- ❌ Emotion-aware coaching

See [Evolution Roadmap](#evolution-roadmap) for how to build these features.

---

## 🏗️ System Architecture

```
┌─────────────┐
│ Microphone  │
└──────┬──────┘
       │ 20ms frames
       ▼
┌─────────────────────┐
│ Feature Extraction  │  ← RMS Energy, Pitch (F0), Speaking Rate
└──────┬──────────────┘
       │ [3 features]
       ▼
┌─────────────────────┐
│ Neural Network      │  ← Lightweight (2K parameters, <5ms inference)
└──────┬──────────────┘
       │ [3 control signals]
       ▼
┌─────────────────────┐
│ DSP Modification    │  ← Pitch shift, Energy boost, Pace adjust
└──────┬──────────────┘
       │ Modified audio
       ▼
┌─────────────┐
│  Speakers   │
└─────────────┘
```

**Latency Budget:**
- Audio I/O buffering: ~40ms (20ms input + 20ms output)
- Feature extraction: ~5-10ms
- Neural network inference: ~5-15ms
- DSP modification: ~10-20ms
- **Total: 60-85ms** ✓ (within <100ms target)

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- Microphone and speakers/headphones

### Setup

```bash
# Clone or navigate to project directory
cd "Tonetta Neural Network"

# Install dependencies
pip install -r requirements.txt

# Note: If pyrubberband fails, you may need to install rubberband:
# macOS: brew install rubberband
# Linux: sudo apt-get install rubberband-cli
```

---

## 🚀 Quick Start

### Test Individual Components

```bash
# Test 1: Audio pipeline (microphone capture + feature extraction)
python examples/test_audio_pipeline.py

# Test 2: Neural network (inference speed)
python examples/test_model_inference.py

# Test 3: DSP modifications (pitch, energy, pace)
python examples/test_dsp_quality.py
```

### Run Complete System

```bash
# Full real-time voice modification
python examples/run_realtime_demo.py

# Speak into your microphone - you'll hear your modified voice!
# Press Ctrl+C to stop and see performance statistics
```

---

## 📚 Understanding the Code (For Non-ML Users)

### What is a Neural Network?

Think of teaching a student to adjust radio knobs (pitch, volume, speed) based on what they hear:

1. **At first**, they turn knobs randomly (bad results)
2. **After seeing examples**, they learn patterns: "When you hear THIS, turn knobs THAT way"
3. **Eventually**, they automatically make good adjustments

A neural network is exactly this: a mathematical student that learns from examples.

### How Data Flows Through the Network

```python
# Step 1: Input (what we measure from your voice)
Input: [rms_energy, pitch_hz, speaking_rate]
       ↓
# Step 2: Hidden Layer 1 (32 "feature detectors")
Each neuron learns a pattern, like "high energy + high pitch = excited"
       ↓
# Step 3: Hidden Layer 2 (16 "pattern combiners")
Combines patterns into higher-level concepts
       ↓
# Step 4: Output (control signals for voice modification)
Output: [pitch_shift, energy_boost, pace_adjustment]
```

### What Each Layer Does

**Dense Layer**: Every input connects to every neuron (fully connected)
- Each neuron computes: `output = activation(weights · input + bias)`
- "Weights" are learned during training (currently random)

**ReLU Activation**: `max(0, x)` - kills negative values, keeps positive
- Why? Lets network learn curves, not just straight lines
- Fast to compute, prevents training problems

**Tanh Activation**: Squashes outputs to [-1, 1] range
- Why? Constrains control signals to useful ranges
- Example: ±5 semitones for pitch shift

### Why Separate Neural Network from DSP?

**Neural Network** = The "brain" that decides "how much" to modify
- Learns patterns from data
- Outputs control signals

**DSP (Digital Signal Processing)** = The "hands" that do the actual work
- Highly optimized C code (fast!)
- Produces high-quality audio
- Can be swapped without retraining network

---

## 🔧 Code Structure

```
Tonetta Neural Network/
├── audio_pipeline.py      # Step 1: Microphone capture + feature extraction
├── voice_model.py         # Step 2: Neural network for control signals
├── audio_modifier.py      # Step 3: DSP-based audio modification
├── realtime_system.py     # Step 4: Complete real-time integration
├── requirements.txt       # Python dependencies
├── examples/              # Test scripts
│   ├── test_audio_pipeline.py
│   ├── test_model_inference.py
│   ├── test_dsp_quality.py
│   └── run_realtime_demo.py
└── README.md             # This file
```

---

## 🎓 Evolution Roadmap

### Phase 1: Foundation (Current) ✓

**What we built:**
- Real-time audio pipeline
- Basic feature extraction
- Lightweight neural network (untrained)
- DSP modification engine

**Why this first:**
- Proves technical feasibility (<100ms latency is achievable)
- Establishes performance baseline
- Creates infrastructure for data collection

---

### Phase 2: Confidence Modeling (Months 2-3)

**Goal:** Detect and boost confidence in real-time

**Technical Additions:**
- Expanded feature set:
  - Spectral features (MFCCs, spectral centroid)
  - Prosody (pitch variance, speech rate variance)
  - Voice quality (jitter, shimmer)
- Larger neural network with temporal context (LSTM/GRU)
- Training data: labeled calls with confidence ratings

**How to Collect Training Data:**
1. **Self-recording**: Record practice calls, rate your own confidence (1-10)
2. **Crowdsourcing**: Use platforms like Amazon MTurk to label call segments
3. **Expert labels**: Partner with speech coaches for high-quality labels
4. **Active learning**: Start with small dataset, iteratively improve

**Model Evolution:**
```python
# Current: Simple feedforward
Input (3 features) → Dense(32) → Dense(16) → Output (3 controls)

# Future: Temporal modeling
Input (10 features × 50 frames) → LSTM(64) → Dense(32) → Output (4 controls + confidence_score)
```

**Training Strategy:**
```python
# Collect data
recordings = []  # List of (audio, confidence_label) pairs

# Train model
model.fit(
    x=audio_features,
    y=desired_control_signals,
    epochs=100,
    validation_split=0.2
)

# Deploy updated weights
model.save('trained_model.h5')
```

---

### Phase 3: Pace Matching (Months 4-5)

**Goal:** Automatically match speaking pace to conversation partner

**Technical Approach:**
- **Dual-stream processing**: Analyze both speakers simultaneously
- **Speaker diarization**: Detect who's speaking (you vs. them)
- **Speaking rate extraction**: Measure syllables per second over 1-2 second windows
- **Adaptive adjustment**: Gradually adjust your pace to match theirs

**Implementation:**
```python
# Detect other speaker
other_speaker_rate = detect_speaking_rate(audio, speaker_id='other')

# Adjust your pace to match
target_pace = smooth_transition(current_pace, other_speaker_rate)
control_signals['pace_multiplier'] = target_pace
```

**Challenges:**
- Need to separate speakers (use voice activity detection + pitch differences)
- Smoothing transitions to avoid jarring changes
- Handling overlapping speech

---

### Phase 4: Emotion-Aware Coaching (Months 6+)

**Goal:** Detect emotional state and provide context-aware modifications

**Technical Additions:**
- **Emotion recognition**: Classify audio into categories (happy, stressed, nervous, confident, calm)
- **Context-aware rules**: Different modifications for different emotions
  - Stressed → calm down (lower pitch, slower pace)
  - Nervous → boost confidence (slight pitch increase, energy boost)
  - Flat → energize (pitch variation, pace increase)

**Model Complexity:**
- Requires larger dataset with emotion labels
- Consider transfer learning from pre-trained models:
  - **wav2vec 2.0**: Pre-trained audio encoder from Facebook AI
  - **HuBERT**: Self-supervised speech representation
- May need GPU for real-time inference at this stage

**Example Architecture:**
```python
# Pre-trained encoder
audio → wav2vec2 → [768-dim embeddings]
                         ↓
# Fine-tuned classifier
                   LSTM(128) → Dense(64) → Softmax(5 emotions)
                         ↓
# Emotion-aware control
                   Rule Engine → Control Signals
```

---

### Phase 5: Production Deployment (Months 6-9)

**Mobile Apps:**
- iOS: Use Core Audio for low-latency audio I/O
- Android: Use AAudio API
- Convert TensorFlow model to TensorFlow Lite for mobile

**Cloud Infrastructure:**
- Model training pipeline (AWS SageMaker, Google Vertex AI)
- A/B testing framework
- User feedback collection
- Model versioning and updates

**Scalability:**
- On-device inference (no cloud latency)
- Cloud-based model updates
- Telemetry for performance monitoring

---

## 🚦 What to Build Now vs. Later

### ✅ Build Now (Weeks 1-4)
- Real-time audio pipeline ✓
- Basic feature extraction ✓
- Simple neural network (even with random weights) ✓
- DSP modification engine ✓
- End-to-end integration ✓
- Latency optimization ✓

### 🔜 Build Soon (Months 2-3)
- Data collection infrastructure
- Training pipeline for neural network
- A/B testing framework
- User feedback collection
- Basic confidence modeling

### ⏳ Build Later (Months 4-6)
- Advanced features (pace matching, emotion)
- Mobile deployment
- Cloud-based model updates
- Multi-language support

### ❌ Don't Build Yet
- Complex emotion models (need data first)
- Real-time transcription (adds latency)
- Multi-speaker scenarios (focus on 1-on-1 calls first)
- Video integration (audio-only is simpler)

---

## 🎯 Key Startup Principles

1. **Start Simple**: Current system gets you to a working demo in 1-2 weeks
2. **Measure Everything**: Latency, quality, user satisfaction
3. **Iterate Based on Data**: Don't guess what features users want
4. **Optimize for Speed**: <100ms latency is your competitive moat
5. **Plan for Scale**: Design system to handle cloud deployment later

---

## 🐛 Troubleshooting

### Audio Issues

**No audio input/output:**
```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Set default device in code:
sd.default.device = [input_device_id, output_device_id]
```

**Audio glitches/dropouts:**
- Increase buffer size (trade latency for stability)
- Close other audio applications
- Check CPU usage (should be <50% for one core)

### Performance Issues

**High latency (>100ms):**
- Profile with `cProfile` to find bottlenecks
- Reduce neural network size
- Use faster pitch detection algorithm
- Consider Cython/Numba for hot paths

**Slow inference:**
- Ensure TensorFlow is using optimized CPU instructions
- Batch process multiple frames when possible
- Consider quantization (int8 instead of float32)

### Installation Issues

**pyrubberband fails:**
```bash
# macOS
brew install rubberband

# Linux
sudo apt-get install rubberband-cli

# Windows
# Download from: https://breakfastquay.com/rubberband/
```

**librosa fails:**
```bash
# Install audio backend
pip install soundfile
```

---

## 📊 Performance Benchmarks

**Expected Performance (on modern laptop):**
- Feature extraction: 5-10ms per frame
- Neural network inference: 2-5ms per frame
- DSP modification: 10-20ms per frame
- **Total processing: 17-35ms per frame**
- **End-to-end latency: 60-85ms** ✓

**If your system is slower:**
- Check CPU usage (close background apps)
- Reduce neural network size
- Use simpler pitch detection
- Profile code to find bottlenecks

---

## 🤝 Contributing

This is a startup project, but contributions are welcome:
- Performance optimizations
- Better feature extraction algorithms
- Training data collection tools
- Mobile deployment guides

---

## 📄 License

[Add your license here]

---

## 🙏 Acknowledgments

**Libraries Used:**
- **sounddevice**: Low-latency audio I/O
- **librosa**: Audio feature extraction and DSP
- **TensorFlow/Keras**: Neural network framework
- **pyrubberband**: High-quality time stretching
- **NumPy**: Numerical computing

---

## 📞 Contact

[Add your contact information]

---

**Built with ❤️ for real-time voice coaching**
