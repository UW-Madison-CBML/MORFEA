#!/bin/bash
# Upload ConvLSTM Autoencoder to GitHub
# Target: https://github.com/UW-Madison-CBML/ivf/Raffael/2025-11-19/

set -e

DATE_FOLDER="2025-11-19"
REPO_URL="git@github.com:UW-Madison-CBML/ivf.git"
TARGET_DIR="Raffael/${DATE_FOLDER}"
SOURCE_DIR="/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

echo "=== Uploading ConvLSTM Autoencoder to GitHub ==="
echo "Repository: ${REPO_URL}"
echo "Target folder: ${TARGET_DIR}"
echo ""

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Clone or update repository
if [ -d "ivf" ]; then
    echo "Repository already exists, updating..."
    cd ivf
    git pull
else
    echo "Cloning repository..."
    git clone ${REPO_URL}
    cd ivf
fi

# Create target directory
echo "Creating directory structure..."
mkdir -p "${TARGET_DIR}"

# Copy essential files only
echo "Copying files..."
cp -v "${SOURCE_DIR}/conv_lstm.py" "${TARGET_DIR}/"
cp -v "${SOURCE_DIR}/model.py" "${TARGET_DIR}/"
cp -v "${SOURCE_DIR}/losses.py" "${TARGET_DIR}/"
cp -v "${SOURCE_DIR}/train.py" "${TARGET_DIR}/"
cp -v "${SOURCE_DIR}/test_model.py" "${TARGET_DIR}/"
cp -v "${SOURCE_DIR}/verify_data_connection.py" "${TARGET_DIR}/"
cp -v "${SOURCE_DIR}/README.md" "${TARGET_DIR}/"

echo ""
echo "Files copied successfully!"
echo ""
echo "Files in ${TARGET_DIR}/:"
ls -lh "${TARGET_DIR}/"
echo ""

# Check git status
echo "Git status:"
git status
echo ""

echo "Next steps:"
echo "1. Review files: git diff"
echo "2. Add files: git add ${TARGET_DIR}/"
echo "3. Commit: git commit -m 'Add ConvLSTM Autoencoder (Raffael/${DATE_FOLDER})'"
echo "4. Push: git push"
echo ""
read -p "Do you want to add, commit, and push now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git add "${TARGET_DIR}/"
    git commit -m "Add ConvLSTM Autoencoder implementation (Raffael/${DATE_FOLDER})"
    git push
    echo ""
    echo "✓ Successfully uploaded to GitHub!"
    echo "View at: https://github.com/UW-Madison-CBML/ivf/tree/main/${TARGET_DIR}"
fi





