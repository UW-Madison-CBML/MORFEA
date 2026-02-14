#!/bin/bash
# Quick upload script for GitHub
# Uploads files to: https://github.com/UW-Madison-CBML/ivf/Raffael/2025-11-19/

set -e

DATE_FOLDER="2025-11-19"
REPO_URL="https://github.com/UW-Madison-CBML/ivf.git"
TARGET_DIR="Raffael/${DATE_FOLDER}"

echo "=== Uploading ConvLSTM Autoencoder to GitHub ==="
echo "Target: ${REPO_URL}"
echo "Folder: ${TARGET_DIR}"
echo ""

# Check if we're in the right directory
if [ ! -f "model.py" ]; then
    echo "Error: model.py not found. Please run this script from Autoencoder_Decoder_ver02 directory."
    exit 1
fi

# Check if git repo exists
if [ ! -d "../../.git" ] && [ ! -d "../../../.git" ]; then
    echo "Git repository not found. Cloning..."
    cd ../..
    git clone ${REPO_URL} ivf-repo
    cd ivf-repo
else
    # Find git root
    GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "")
    if [ -z "$GIT_ROOT" ]; then
        echo "Error: Not in a git repository. Please clone the repo first."
        exit 1
    fi
    cd "$GIT_ROOT"
fi

# Create target directory
mkdir -p "${TARGET_DIR}"

# Copy essential files only
echo "Copying files..."
cp -v ../Code/Autoencoder_Decoder_ver02/conv_lstm.py "${TARGET_DIR}/"
cp -v ../Code/Autoencoder_Decoder_ver02/model.py "${TARGET_DIR}/"
cp -v ../Code/Autoencoder_Decoder_ver02/losses.py "${TARGET_DIR}/"
cp -v ../Code/Autoencoder_Decoder_ver02/train.py "${TARGET_DIR}/"
cp -v ../Code/Autoencoder_Decoder_ver02/test_model.py "${TARGET_DIR}/"
cp -v ../Code/Autoencoder_Decoder_ver02/verify_data_connection.py "${TARGET_DIR}/"
cp -v ../Code/Autoencoder_Decoder_ver02/README.md "${TARGET_DIR}/"

echo ""
echo "Files copied to ${TARGET_DIR}/"
echo ""
echo "Next steps:"
echo "1. Review the files: git status"
echo "2. Add files: git add ${TARGET_DIR}/"
echo "3. Commit: git commit -m 'Add ConvLSTM Autoencoder (Raffael/${DATE_FOLDER})'"
echo "4. Push: git push"
echo ""
echo "Or run: git add ${TARGET_DIR}/ && git commit -m 'Add ConvLSTM Autoencoder (Raffael/${DATE_FOLDER})' && git push"





