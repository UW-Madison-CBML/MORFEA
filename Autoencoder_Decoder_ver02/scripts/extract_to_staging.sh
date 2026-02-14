#!/bin/bash
# Extract embryo_dataset.tar.gz directly to staging directory
# This avoids home directory quota limits

set -e  # Exit on error

echo "============================================================"
echo "Extract Dataset to Staging Directory"
echo "============================================================"
echo ""

# Define paths
TAR_FILE="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"
STAGING_DIR="/staging/groups/bhaskar_group/rho9/ivf_data"
EXTRACT_DIR="$STAGING_DIR/embryo_dataset"

# Check if tar.gz exists
if [ ! -f "$TAR_FILE" ]; then
    echo "❌ ERROR: Tar.gz file not found:"
    echo "   $TAR_FILE"
    echo ""
    echo "Please check the file location."
    exit 1
fi

# Check tar.gz size
TAR_SIZE=$(du -sh "$TAR_FILE" | cut -f1)
echo "✓ Found tar.gz file: $TAR_SIZE"
echo ""

# Check if extraction directory already exists
if [ -d "$EXTRACT_DIR" ]; then
    EXISTING_SIZE=$(du -sh "$EXTRACT_DIR" 2>/dev/null | cut -f1)
    EXISTING_COUNT=$(ls "$EXTRACT_DIR" 2>/dev/null | wc -l)
    echo "⚠️  Extraction directory already exists:"
    echo "   $EXTRACT_DIR"
    echo "   Current size: $EXISTING_SIZE"
    echo "   Cell directories: $EXISTING_COUNT"
    echo ""
    read -p "Do you want to remove it and re-extract? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing directory..."
        rm -rf "$EXTRACT_DIR"
        echo "✓ Removed"
    else
        echo "Skipping extraction. Using existing directory."
        exit 0
    fi
fi

# Check staging directory space
echo "Checking staging directory space..."
STAGING_AVAIL=$(df -h "$STAGING_DIR" | tail -1 | awk '{print $4}')
echo "   Available space: $STAGING_AVAIL"
echo ""

# Extract to staging
echo "============================================================"
echo "Extracting to staging directory..."
echo "============================================================"
echo "Source: $TAR_FILE"
echo "Destination: $EXTRACT_DIR"
echo ""
echo "⚠️  This will take 10-30 minutes depending on disk speed"
echo ""

# Change to staging directory and extract
cd "$STAGING_DIR"
echo "Extracting..."
tar -xzvf "$TAR_FILE"

echo ""
echo "============================================================"
echo "✅ Extraction Complete!"
echo "============================================================"
echo ""

# Verify extraction
if [ -d "$EXTRACT_DIR" ]; then
    FINAL_SIZE=$(du -sh "$EXTRACT_DIR" | cut -f1)
    FINAL_COUNT=$(ls "$EXTRACT_DIR" | wc -l)
    echo "Extracted directory: $EXTRACT_DIR"
    echo "Size: $FINAL_SIZE"
    echo "Cell directories: $FINAL_COUNT"
    echo ""
    echo "First 10 cells:"
    ls "$EXTRACT_DIR" | head -10
    echo ""
    echo "✅ Dataset ready for use!"
else
    echo "❌ ERROR: Extraction directory not found after extraction"
    exit 1
fi

