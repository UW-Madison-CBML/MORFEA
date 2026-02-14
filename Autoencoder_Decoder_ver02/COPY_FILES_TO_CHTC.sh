#!/bin/bash
# Script to copy all necessary files to ~/ivf_train on CHTC

echo "=== Copying files to ~/ivf_train ==="

# Navigate to training directory
cd ~/ivf_train || { echo "ERROR: ~/ivf_train not found!"; exit 1; }

# Find the repo location
REPO_DIR=""
if [ -d ~/ivf/Raffael/2025-11-19 ]; then
    REPO_DIR=~/ivf/Raffael/2025-11-19
elif [ -d ~/ivf/Code/Autoencoder_Decoder_ver02 ]; then
    REPO_DIR=~/ivf/Code/Autoencoder_Decoder_ver02
elif [ -d ~/ivf-embryo-analysis-Raffael ]; then
    REPO_DIR=~/ivf-embryo-analysis-Raffael
else
    echo "ERROR: Cannot find repo directory!"
    echo "Please run: find ~ -name 'build_index.py' -type f 2>/dev/null | head -5"
    exit 1
fi

echo "Found repo at: $REPO_DIR"

# Copy files
echo "Copying files..."
cp "$REPO_DIR/run_train.sh" . && echo "✓ run_train.sh"
cp "$REPO_DIR/train_h200_lab.sub" . && echo "✓ train_h200_lab.sub"
cp "$REPO_DIR/build_index.py" . && echo "✓ build_index.py"
cp "$REPO_DIR/train.py" . && echo "✓ train.py"
cp "$REPO_DIR/model.py" . && echo "✓ model.py"
cp "$REPO_DIR/conv_lstm.py" . && echo "✓ conv_lstm.py"
cp "$REPO_DIR/losses.py" . && echo "✓ losses.py"

# Find and copy dataset_ivf.py
if [ -f ~/ivf/dataset_ivf.py ]; then
    cp ~/ivf/dataset_ivf.py . && echo "✓ dataset_ivf.py"
elif [ -f "$REPO_DIR/../dataset_ivf.py" ]; then
    cp "$REPO_DIR/../dataset_ivf.py" . && echo "✓ dataset_ivf.py"
else
    echo "⚠ WARNING: dataset_ivf.py not found! Searching..."
    find ~/ivf -name "dataset_ivf.py" -type f 2>/dev/null | head -1 | xargs -I {} cp {} . && echo "✓ dataset_ivf.py (found)"
fi

echo ""
echo "=== Files in ~/ivf_train ==="
ls -lh *.py *.sh *.sub 2>/dev/null | awk '{print $9, "(" $5 ")"}'

echo ""
echo "=== Ready to submit ==="
echo "Run: condor_submit train_h200_lab.sub"





