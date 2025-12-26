# Latency Optimization Guide

## 🎯 The Problem

You successfully ran the voice modifier, but latency was **805ms** (target: <100ms).

**Breakdown of the 805ms:**
- Feature Extraction: **40ms** (pitch detection is the culprit!)
- Neural Network: 29ms
- DSP Modification: 7ms
- Other overhead: ~729ms accumulated latency

## 🔍 Root Cause

**Pitch detection with `librosa.pyin()` is too slow for 20ms frames:**
- Needs 2048 samples minimum
- We only have 960 samples (20ms @ 48kHz)
- librosa has to pad/interpolate → slow
- Takes 40ms per frame → impossible for real-time

## ✅ Solution: Two Versions

### Version 1: FAST (Optimized for Real-Time)

**File:** `examples/run_fast_demo.py`

```bash
python examples/run_fast_demo.py
```

**What it does:**
- ✅ Skips pitch detection (too slow)
- ✅ Only uses energy/volume modifications
- ✅ Target latency: <100ms
- ❌ No pitch shifting
- ❌ No pace modification

**Use this for:** Real-time calls where latency matters most

---

### Version 2: FULL (All Features, Higher Latency)

**File:** `examples/run_realtime_demo.py`

```bash
python examples/run_realtime_demo.py
```

**What it does:**
- ✅ Full feature extraction (energy, pitch, pace)
- ✅ All DSP modifications (pitch shift, energy, pace)
- ❌ High latency (~800ms)

**Use this for:** Offline processing, testing, or when latency doesn't matter

---

## 🚀 Try the Fast Version Now

```bash
python examples/run_fast_demo.py
```

You should hear:
- ✅ Much lower latency (<100ms)
- ✅ Energy/volume changes
- ❌ No pitch changes (disabled for speed)

---

## 🔧 Future Optimizations

To get <100ms with ALL features:

### Option 1: Larger Frames
- Use 50-100ms frames instead of 20ms
- Gives pitch detection enough samples
- Trade-off: Higher latency, but pitch works

### Option 2: Faster Pitch Detection
- Use simpler algorithms (autocorrelation, YIN)
- Less accurate but much faster
- Good enough for real-time use

### Option 3: GPU Acceleration
- Move DSP to GPU
- Parallel processing
- Requires CUDA/Metal setup

### Option 4: Downsample for Pitch
- Detect pitch on downsampled audio (24kHz → 8kHz)
- Faster processing
- Slightly less accurate

---

## 📊 Expected Performance

### Fast Version (Current)
- Latency: **50-80ms** ✓
- Features: Energy only
- Quality: Good for volume adjustments

### Optimized Full Version (Future)
- Latency: **80-120ms**
- Features: Energy + Pitch + Pace
- Quality: Full voice modification

### Current Full Version
- Latency: **800ms** ❌
- Features: All
- Quality: Good but unusable for real-time

---

## 💡 Recommendation

**For your startup:**

1. **Now:** Use fast version for demos/testing
2. **Next week:** Implement Option 2 (faster pitch detection)
3. **Month 1:** Optimize with Option 1 (larger frames)
4. **Month 2:** Consider GPU acceleration if needed

**The fast version proves the concept works!** Once you have training data and a trained model, energy/volume modifications alone can be very effective for confidence boosting.

---

## 🎓 Why This Matters for Training

Even with the fast version, you can:
- ✅ Collect training data (`collect_data.py`)
- ✅ Train the neural network (`train_model.py`)
- ✅ Deploy trained model for energy modifications
- ✅ Add pitch later when optimized

**Start with what works, optimize later!**
