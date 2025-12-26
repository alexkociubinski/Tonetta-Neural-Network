# How to Run and Train Tonetta

## 🚀 Quick Start (3 Commands)

### 1. Install Dependencies

```bash
cd "Tonetta Neural Network"
./setup.sh
```

This creates a virtual environment and installs all dependencies.

### 2. Test the System

```bash
# Activate virtual environment
source venv/bin/activate

# Run the complete system
python examples/run_realtime_demo.py
```

**What happens:**
- Microphone captures your voice
- Features are extracted (energy, pitch, speaking rate)
- Neural network generates control signals (currently random)
- DSP modifies your voice
- You hear the modified voice through speakers

**Press Ctrl+C to stop and see performance statistics**

### 3. Check Performance

You should see:
- ✅ Latency: 60-85ms (below 100ms target)
- ✅ Frame rate: ~50 fps
- ✅ No audio glitches

---

## 🎓 Training the Model (Step-by-Step)

Right now the neural network has **random weights** (not trained), so modifications won't be meaningful. Here's how to train it:

### Step 1: Collect Training Data

```bash
source venv/bin/activate
python collect_data.py
```

**What this does:**
1. Records you speaking for 30 seconds
2. Asks you to rate:
   - Confidence (1-10): How confident did you sound?
   - Energy (1-10): How energetic were you?
   - Pace (1-10): How fast were you speaking?
3. Saves recording + labels to `training_data/`
4. Repeat 10-20 times

**Tips for good training data:**
- Vary your speaking style (confident, nervous, energetic, calm)
- Speak naturally, as if on a real call
- Be honest with your ratings
- Record in a quiet environment

### Step 2: Train the Model

```bash
source venv/bin/activate
python train_model.py
```

**What this does:**
1. Loads your recordings from `training_data/`
2. Extracts features from each recording
3. Trains neural network to learn patterns:
   - "When confidence is low → boost pitch/energy"
   - "When energy is high → maintain current levels"
4. Saves trained model to `models/trained_model.h5`

**Training time:** 5-10 minutes on laptop CPU

**What you'll see:**
```
Training for 50 epochs...
Epoch 1/50
loss: 0.234 - val_loss: 0.198
Epoch 2/50
loss: 0.187 - val_loss: 0.165
...
✓ Training complete!
```

### Step 3: Use Trained Model

Edit `realtime_system.py` to load your trained model:

```python
# Around line 50, change:
self.voice_model = VoiceControlModel()

# To:
self.voice_model = VoiceControlModel()
self.voice_model.load('models/trained_model.h5')
```

Then run:
```bash
python examples/run_realtime_demo.py
```

Now the modifications should match your training data!

---

## 📊 Understanding Training

### What is Training?

Think of the neural network as a student learning to adjust radio knobs:

**Before Training (Current):**
- Student turns knobs randomly
- Results are meaningless
- No pattern to the adjustments

**After Training:**
- Student has seen 1000+ examples
- Learned: "When voice sounds nervous → boost confidence"
- Makes intelligent adjustments

### How Training Works

1. **Show examples:** "This voice (features) should get these adjustments (targets)"
2. **Model predicts:** Network makes a guess
3. **Calculate error:** How wrong was the guess?
4. **Adjust weights:** Tweak network to reduce error
5. **Repeat:** Go through all examples many times (epochs)

### What Makes Good Training Data?

**Quantity:**
- Minimum: 10 recordings (for testing)
- Good: 50-100 recordings
- Production: 1000+ recordings

**Quality:**
- Consistent labeling (be honest with ratings)
- Variety (different moods, speaking styles)
- Clean audio (minimal background noise)

**Diversity:**
- Different confidence levels (1-10)
- Different energy levels (calm to excited)
- Different paces (slow to fast)

---

## 🔍 Testing Your Trained Model

### Quick Test

```bash
python examples/run_realtime_demo.py
```

Speak with different confidence levels and listen for changes.

### What to Listen For

**If training worked:**
- Low confidence voice → pitch/energy boost
- High confidence voice → minimal changes
- Smooth, natural-sounding modifications

**If training didn't work:**
- Random or no modifications
- Distorted audio
- Inconsistent behavior

**Common issues:**
- Not enough training data (need 20+ recordings)
- Inconsistent labeling (ratings don't match audio)
- Overfitting (works on training data, not new audio)

---

## 🎯 Next Steps After Training

### Immediate
1. ✅ Collect 20+ training recordings
2. ✅ Train your first model
3. ✅ Test on live audio
4. ✅ Collect feedback from friends

### Short-Term (Weeks 2-4)
1. Collect 100+ recordings
2. Experiment with different architectures
3. Add more features (MFCCs, prosody)
4. Improve training data quality

### Medium-Term (Months 2-3)
1. Build confidence detection
2. Add real-time feedback UI
3. Implement pace matching
4. Test with beta users

---

## 💡 Pro Tips

### For Better Training
- **More data = better model** (aim for 100+ hours eventually)
- **Consistent labeling** is more important than quantity
- **Start simple** (just confidence), add complexity later
- **Monitor validation loss** to prevent overfitting

### For Better Real-Time Performance
- Close other audio applications
- Use wired headphones (less latency than Bluetooth)
- Keep CPU usage <50%
- Test on different hardware

### For Production
- Collect user feedback continuously
- A/B test different models
- Monitor latency in production
- Iterate based on real usage data

---

## 🐛 Troubleshooting Training

### "Not enough training data"
- Need at least 10 recordings to start
- Run `collect_data.py` to collect more

### "Training loss not decreasing"
- Check if labels are consistent
- Try lower learning rate (0.0001 instead of 0.001)
- Collect more diverse data

### "Model works on training data but not live audio"
- Overfitting - need more training data
- Add dropout layers to model
- Use data augmentation

### "Modifications sound distorted"
- DSP parameters might be too extreme
- Reduce output ranges in voice_model.py
- Check audio quality of training data

---

## 📈 Measuring Success

### Technical Metrics
- **Latency:** <100ms ✓
- **Training loss:** <0.1 (after convergence)
- **Validation loss:** Similar to training loss

### User Metrics
- **Does it sound natural?** (most important)
- **Do modifications match intent?**
- **Would you use this on real calls?**

---

## 🎓 Learning Resources

### Understanding Neural Networks
- See `voice_model.py` for plain English explanations
- Each layer is documented with what it does
- Training process explained step-by-step

### Understanding Audio Processing
- See `audio_pipeline.py` for feature extraction
- See `audio_modifier.py` for DSP techniques
- All concepts explained for non-ML users

---

**Ready to start?** Run `./setup.sh` and begin testing!
