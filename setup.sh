#!/bin/bash
# Setup script for STT project

echo "Setting up Speech to Text with Whisper..."

# Check if Python 3.11+ is available
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo "Found Python version: $PYTHON_VERSION"
    
    # Check if version is 3.11 or higher
    if [[ $(printf '%s\n' "3.11" "$PYTHON_VERSION" | sort -V | head -n1) == "3.11" ]] || [[ "$PYTHON_VERSION" == "3.11" ]]; then
        echo "Python version is compatible."
    else
        echo "Warning: Python 3.11 or higher is recommended."
    fi
else
    echo "Python3 not found. Please install Python 3.11 or higher."
    exit 1
fi

# Check if uv is available, install if not
if ! command -v uv &>/dev/null; then
    echo "uv not found. Installing uv..."
    pip install uv
fi

# Install dependencies with uv
echo "Installing dependencies with uv..."
uv pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "To transcribe audio, run:"
echo "python main.py <audio_file_path>"
echo ""
echo "For help, run:"
echo "python main.py --help"