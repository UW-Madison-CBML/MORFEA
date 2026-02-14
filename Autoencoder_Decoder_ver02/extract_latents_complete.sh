#!/bin/bash
# extract_latents.sh
# Wrapper script for extracting latent trajectories on CHTC
# This script copies files from staging to working directory

set -e

echo "=== Extract Latent Trajectories on CHTC ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Staging directory
STAGING_DIR="/staging/groups/bhaskar_group/rho9"

# Parse arguments
CHECKPOINT_ARG="$1"
MODEL_VERSION="$2"

if [ -z "$CHECKPOINT_ARG" ] || [ -z "$MODEL_VERSION" ]; then
    echo "Usage: $0 <checkpoint_path> <model_version_name>"
    exit 1
fi

echo "Checkpoint argument: $CHECKPOINT_ARG"
echo "Model version: $MODEL_VERSION"
echo ""

# Copy files from staging to working directory
echo "Copying files from staging to working directory..."
cp "$STAGING_DIR/extract_all_latent_trajectories.py" ./
cp "$STAGING_DIR/model.py" ./
cp "$STAGING_DIR/dataset_ivf.py" ./
cp "$STAGING_DIR/build_index.py" ./
cp "$STAGING_DIR/index.csv" ./

# Handle checkpoint path
if [[ "$CHECKPOINT_ARG" == /* ]]; then
    # Absolute path - copy from staging
    if [ ! -f "$CHECKPOINT_ARG" ]; then
        echo "Error: Checkpoint not found: $CHECKPOINT_ARG"
        exit 1
    fi
    cp "$CHECKPOINT_ARG" ./checkpoint.pt
    CHECKPOINT="./checkpoint.pt"
else
    # Relative path
    CHECKPOINT="$CHECKPOINT_ARG"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint not found: $CHECKPOINT"
        exit 1
    fi
fi

echo "✓ Files copied successfully"
echo ""

# Setup Python environment
echo "Setting up Python environment..."
python3 --version

# Check and install missing packages (install all at once to avoid loops)
echo "Checking Python packages..."
python3 << 'PYTHON_EOF'
import sys
import subprocess

missing = []
try:
    import torch
    print("  ✓ torch")
except ImportError:
    missing.append("torch")
    print("  ✗ torch - MISSING")

try:
    import numpy
    print("  ✓ numpy")
except ImportError:
    missing.append("numpy")
    print("  ✗ numpy - MISSING")

try:
    import pandas
    print("  ✓ pandas")
except ImportError:
    missing.append("pandas")
    print("  ✗ pandas - MISSING")

try:
    import PIL
    print("  ✓ Pillow (PIL)")
except ImportError:
    missing.append("pillow")
    print("  ✗ Pillow (PIL) - MISSING")

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    print("Installing all packages at once (this may take a few minutes)...")
    try:
        # Install all at once - faster and avoids dependency conflicts
        cmd = [sys.executable, "-m", "pip", "install", "--user", "--no-cache-dir"] + missing
        result = subprocess.run(cmd, timeout=600, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ All packages installed successfully")
        else:
            print(f"✗ Installation failed: {result.stderr}")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("✗ Installation timed out (>10 minutes)")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Installation error: {e}")
        sys.exit(1)
else:
    print("✓ All required packages available")
PYTHON_EOF

# Check if index.csv exists
if [ ! -f "index.csv" ]; then
    echo "Error: index.csv not found after copy"
    exit 1
fi

# Determine device
DEVICE="cpu"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        DEVICE="cuda"
        echo "GPU detected, using CUDA"
        nvidia-smi
    fi
else
    echo "No GPU detected, using CPU"
fi

# Run extraction
echo ""
echo "Running extraction..."
MAX_EMBRYOS=""  # Set to number like "10" to limit, or "" for all

# Check if files exist before running
echo "Checking files..."
ls -la extract_all_latent_trajectories.py model.py dataset_ivf.py build_index.py index.csv
echo ""

# Run with error handling
set +e
if [ -n "$MAX_EMBRYOS" ]; then
    python3 extract_all_latent_trajectories.py \
        --checkpoint "$CHECKPOINT" \
        --model_version "$MODEL_VERSION" \
        --index_csv index.csv \
        --device "$DEVICE" \
        --max_embryos "$MAX_EMBRYOS" \
        --output_dir model_latents 2>&1
    EXIT_CODE=$?
else
    python3 extract_all_latent_trajectories.py \
        --checkpoint "$CHECKPOINT" \
        --model_version "$MODEL_VERSION" \
        --index_csv index.csv \
        --device "$DEVICE" \
        --output_dir model_latents 2>&1
    EXIT_CODE=$?
fi
set -e

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Python script failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "=== Extraction Complete ==="
if [ -d "model_latents/$MODEL_VERSION/" ]; then
    ls -lh "model_latents/$MODEL_VERSION/"
fi

