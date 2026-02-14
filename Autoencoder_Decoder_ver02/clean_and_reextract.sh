#!/bin/bash
# Remove incomplete extraction (RI382-2 folder) and re-extract complete dataset

echo "=== Clean and Re-extract Dataset ==="
echo ""

EXTRACTED_DIR="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset"
TAR_FILE="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"

# Check current state
if [ -d "$EXTRACTED_DIR" ]; then
    echo "Current extracted directory:"
    SIZE=$(du -sh "$EXTRACTED_DIR" 2>/dev/null | cut -f1)
    CELL_COUNT=$(find "$EXTRACTED_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "  Size: $SIZE"
    echo "  Cell directories: $CELL_COUNT"
    echo ""
    
    if [ "$CELL_COUNT" -eq 1 ]; then
        echo "⚠️  Only 1 cell directory found (likely incomplete extraction)"
        echo "   Found: $(find "$EXTRACTED_DIR" -mindepth 1 -maxdepth 1 -type d | head -1 | xargs basename)"
    fi
fi

echo ""
echo "This will:"
echo "  1. Remove incomplete extraction: $EXTRACTED_DIR"
echo "  2. Re-extract from: $TAR_FILE"
echo ""

# Check tar.gz exists
if [ ! -f "$TAR_FILE" ]; then
    echo "❌ Error: Tar.gz file not found: $TAR_FILE"
    exit 1
fi

TAR_SIZE=$(du -sh "$TAR_FILE" | cut -f1)
echo "Tar.gz size: $TAR_SIZE"
echo ""

read -p "Continue? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Cancelled"
    exit 0
fi

# Remove incomplete extraction
echo ""
echo "Step 1: Removing incomplete extraction..."
if [ -d "$EXTRACTED_DIR" ]; then
    rm -rf "$EXTRACTED_DIR"
    echo "✅ Removed: $EXTRACTED_DIR"
else
    echo "   (Directory doesn't exist, skipping)"
fi

# Also remove RI382-2 if it exists as a separate folder
RI382_DIR="/staging/groups/bhaskar_group/rho9/ivf_data/RI382-2"
if [ -d "$RI382_DIR" ]; then
    echo "Removing: $RI382_DIR"
    rm -rf "$RI382_DIR"
    echo "✅ Removed: $RI382_DIR"
fi

echo ""
echo "Step 2: Re-extracting complete dataset..."
echo "   This may take 10-30 minutes..."
echo "   Logging to: extraction.log"
echo ""

cd /staging/groups/bhaskar_group/rho9/ivf_data

# Extract with verbose output
tar -xzvf embryo_dataset.tar.gz 2>&1 | tee extraction.log

EXTRACT_STATUS=$?

echo ""
echo "=== Extraction Complete ==="
echo ""

if [ $EXTRACT_STATUS -eq 0 ]; then
    if [ -d "$EXTRACTED_DIR" ]; then
        EXTRACTED_SIZE=$(du -sh "$EXTRACTED_DIR" | cut -f1)
        EXTRACTED_SIZE_BYTES=$(du -sb "$EXTRACTED_DIR" | cut -f1)
        EXTRACTED_SIZE_GB=$(echo "scale=2; $EXTRACTED_SIZE_BYTES / 1024 / 1024 / 1024" | bc)
        EXTRACTED_CELLS=$(find "$EXTRACTED_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        EXTRACTED_FILES=$(find "$EXTRACTED_DIR" -type f 2>/dev/null | wc -l)
        
        echo "✅ Extraction successful!"
        echo "   Size: $EXTRACTED_SIZE (${EXTRACTED_SIZE_GB} GB)"
        echo "   Cell directories: $EXTRACTED_CELLS"
        echo "   Files: $EXTRACTED_FILES"
        echo ""
        
        if (( $(echo "$EXTRACTED_SIZE_GB > 10" | bc -l) )); then
            echo "✅ Size looks correct (~12 GB expected)"
        else
            echo "⚠️  Size seems small (${EXTRACTED_SIZE_GB} GB, expected ~12 GB)"
            echo "   Check extraction.log for errors"
        fi
        
        # Update symlink
        echo ""
        echo "Updating symlink in ~/ivf_repo..."
        cd ~/ivf_repo 2>/dev/null || cd "$HOME"
        if [ -d "ivf_repo" ]; then
            cd ivf_repo
            rm -f data
            ln -s "$EXTRACTED_DIR" data
            echo "✅ Symlink updated: data -> $EXTRACTED_DIR"
            ls -la data
        fi
    else
        echo "⚠️  Extraction completed but embryo_dataset directory not found"
        echo "   Check extraction.log for details"
    fi
else
    echo "❌ Extraction failed with status: $EXTRACT_STATUS"
    echo "   Check extraction.log for error details"
fi

echo ""
echo "Check extraction.log for detailed output"

