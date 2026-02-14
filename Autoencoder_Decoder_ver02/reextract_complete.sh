#!/bin/bash
# Re-extract the complete dataset from tar.gz
# This will properly extract all 12 GB of data

echo "=== Re-extracting Complete Dataset ==="
echo ""

TAR_FILE="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"
EXTRACT_DIR="/staging/groups/bhaskar_group/rho9/ivf_data"
WORK_DIR="$HOME/ivf_repo/data_raw"

# Check tar.gz exists
if [ ! -f "$TAR_FILE" ]; then
    echo "❌ Error: Tar.gz file not found: $TAR_FILE"
    exit 1
fi

echo "Tar.gz file: $TAR_FILE"
TAR_SIZE=$(du -sh "$TAR_FILE" | cut -f1)
echo "Size: $TAR_SIZE"
echo ""

# Check disk space
echo "Checking available disk space..."
AVAILABLE=$(df -h "$HOME" | tail -1 | awk '{print $4}')
echo "Available space in home: $AVAILABLE"
echo ""

# Remove old incomplete extraction
if [ -d "$EXTRACT_DIR/embryo_dataset" ]; then
    echo "Removing old incomplete extraction..."
    echo "  Current size: $(du -sh "$EXTRACT_DIR/embryo_dataset" | cut -f1)"
    read -p "  Delete old extraction? (y/n): " confirm
    if [ "$confirm" = "y" ]; then
        rm -rf "$EXTRACT_DIR/embryo_dataset"
        echo "  ✓ Removed"
    else
        echo "  Skipped (will extract to different location)"
        WORK_DIR="$HOME/ivf_repo/data_raw"
    fi
fi

echo ""

# Create extraction directory
if [ "$WORK_DIR" = "$HOME/ivf_repo/data_raw" ]; then
    echo "Extracting to: $WORK_DIR"
    mkdir -p "$WORK_DIR"
    EXTRACT_TO="$WORK_DIR"
else
    echo "Extracting to: $EXTRACT_DIR"
    EXTRACT_TO="$EXTRACT_DIR"
fi

echo ""

# Extract with verbose output and error logging
echo "Starting extraction (this may take 10-30 minutes)..."
echo "Logging to: extraction.log"
echo ""

cd "$(dirname "$EXTRACT_TO")"

# Extract with progress and error logging
tar -xzvf "$TAR_FILE" -C "$(dirname "$EXTRACT_TO")" 2>&1 | tee extraction.log

EXTRACT_STATUS=$?

echo ""
echo "=== Extraction Complete ==="
echo ""

if [ $EXTRACT_STATUS -eq 0 ]; then
    # Check extracted size
    if [ -d "$EXTRACT_TO/embryo_dataset" ]; then
        EXTRACTED_SIZE=$(du -sh "$EXTRACT_TO/embryo_dataset" | cut -f1)
        EXTRACTED_CELLS=$(find "$EXTRACT_TO/embryo_dataset" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        EXTRACTED_FILES=$(find "$EXTRACT_TO/embryo_dataset" -type f 2>/dev/null | wc -l)
        
        echo "✅ Extraction successful!"
        echo "   Size: $EXTRACTED_SIZE"
        echo "   Cell directories: $EXTRACTED_CELLS"
        echo "   Files: $EXTRACTED_FILES"
        echo ""
        
        # Check if size is reasonable (should be ~12 GB)
        EXTRACTED_SIZE_BYTES=$(du -sb "$EXTRACT_TO/embryo_dataset" 2>/dev/null | cut -f1)
        EXTRACTED_SIZE_GB=$(echo "scale=2; $EXTRACTED_SIZE_BYTES / 1024 / 1024 / 1024" | bc)
        
        if (( $(echo "$EXTRACTED_SIZE_GB > 10" | bc -l) )); then
            echo "✅ Size looks correct (~${EXTRACTED_SIZE_GB} GB)"
        else
            echo "⚠️  Size seems small (${EXTRACTED_SIZE_GB} GB, expected ~12 GB)"
            echo "   Check extraction.log for errors"
        fi
        
        # Update symlink if extracting to staging
        if [ "$EXTRACT_TO" = "$EXTRACT_DIR/embryo_dataset" ]; then
            echo ""
            echo "Updating symlink..."
            cd "$HOME/ivf_repo"
            rm -f data
            ln -s "$EXTRACT_TO" data
            echo "✓ Symlink updated: data -> $EXTRACT_TO"
        else
            echo ""
            echo "To use this dataset, create symlink:"
            echo "  cd ~/ivf_repo"
            echo "  rm -f data"
            echo "  ln -s $EXTRACT_TO/embryo_dataset data"
        fi
    else
        echo "⚠️  Extraction completed but embryo_dataset directory not found"
        echo "   Check extraction.log for details"
    fi
else
    echo "❌ Extraction failed with status: $EXTRACT_STATUS"
    echo "   Check extraction.log for error details"
    echo ""
    echo "Common issues:"
    echo "  - Disk space full"
    echo "  - Permission denied"
    echo "  - Corrupted tar.gz file"
fi

echo ""
echo "Check extraction.log for detailed output"

