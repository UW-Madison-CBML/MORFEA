#!/bin/bash
# Fix and download all files correctly from GitHub

set -e

echo "=== Downloading ConvLSTM Autoencoder Files ==="
echo ""

cd ~/ivf_train

# GitHub raw URL base
BASE_URL="https://raw.githubusercontent.com/UW-Madison-CBML/ivf/main/Raffael/2025-11-19"

echo "Downloading files from GitHub..."
echo ""

# Remove corrupted files first
rm -f train.py model.py conv_lstm.py losses.py run_train.sh train_h200_lab.sub

# Download Python files
echo "1. Downloading Python files..."
curl -L -o conv_lstm.py "${BASE_URL}/conv_lstm.py"
curl -L -o model.py "${BASE_URL}/model.py"
curl -L -o losses.py "${BASE_URL}/losses.py"
curl -L -o train.py "${BASE_URL}/train.py"

# Download scripts
echo "2. Downloading scripts..."
curl -L -o run_train.sh "${BASE_URL}/run_train.sh"
curl -L -o train_h200_lab.sub "${BASE_URL}/train_h200_lab.sub"

# Verify file sizes (should be > 100 bytes, not 14)
echo ""
echo "3. Verifying files..."
for file in conv_lstm.py model.py losses.py train.py run_train.sh train_h200_lab.sub; do
    SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
    if [ "$SIZE" -lt 100 ]; then
        echo "❌ $file is too small ($SIZE bytes) - download may have failed"
    else
        echo "✅ $file: $SIZE bytes"
    fi
done

# Make executable
echo ""
echo "4. Setting permissions..."
chmod +x run_train.sh
mkdir -p logs

echo ""
echo "=== Files Ready ==="
echo ""
echo "Files:"
ls -lh train.py model.py conv_lstm.py losses.py run_train.sh train_h200_lab.sub
echo ""
echo "Now submit:"
echo "  condor_submit train_h200_lab.sub"





