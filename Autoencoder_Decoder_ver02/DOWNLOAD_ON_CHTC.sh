#!/bin/bash
# Copy and paste this into CHTC terminal
# Downloads files directly from GitHub (no auth needed for public repo)

set -e

echo "=== Downloading ConvLSTM Autoencoder from GitHub ==="

# Create directory
mkdir -p ~/ivf_train
cd ~/ivf_train

# GitHub raw URL base
BASE_URL="https://raw.githubusercontent.com/UW-Madison-CBML/ivf/main/Raffael/2025-11-19"

echo "Downloading files..."

# Download Python files
curl -L -o conv_lstm.py "${BASE_URL}/conv_lstm.py"
curl -L -o model.py "${BASE_URL}/model.py"
curl -L -o losses.py "${BASE_URL}/losses.py"
curl -L -o train.py "${BASE_URL}/train.py"

# Download scripts
curl -L -o run_train.sh "${BASE_URL}/run_train.sh"
curl -L -o train_h200_lab.sub "${BASE_URL}/train_h200_lab.sub"

# Make executable
chmod +x run_train.sh
mkdir -p logs

echo "✓ Files downloaded!"
echo ""
echo "Files:"
ls -lh
echo ""
echo "Next steps:"
echo "  condor_submit train_h200_lab.sub"
echo "  condor_q"





