#!/bin/bash

# Tonetta Setup Script
# This script sets up a virtual environment and installs all dependencies

echo "=========================================="
echo "TONETTA SETUP"
echo "=========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."
echo ""

pip install numpy>=1.24.0
pip install sounddevice>=0.4.6
pip install librosa>=0.10.0
pip install tensorflow>=2.13.0
pip install scipy>=1.10.0

# Try to install pyrubberband (may fail if rubberband not installed)
echo ""
echo "Installing pyrubberband (optional, for time-stretching)..."
pip install pyrubberband>=0.3.0

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  pyrubberband installation failed"
    echo "This is optional but recommended for better audio quality"
    echo ""
    echo "To install rubberband library:"
    echo "  macOS: brew install rubberband"
    echo "  Linux: sudo apt-get install rubberband-cli"
    echo ""
    echo "Then run: pip install pyrubberband"
    echo ""
fi

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "To activate the virtual environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To test the system:"
echo "  python examples/test_audio_pipeline.py"
echo "  python examples/test_model_inference.py"
echo "  python examples/run_realtime_demo.py"
echo ""
echo "To collect training data:"
echo "  python collect_data.py"
echo ""
echo "To train the model:"
echo "  python train_model.py"
echo ""
echo "See QUICKSTART.md for detailed instructions"
echo "=========================================="
