#!/bin/bash
# Re-extract the dataset from tar.gz

echo "=== Re-extracting Dataset ==="
echo ""

BASE_DIR="/staging/groups/bhaskar_group/rho9/ivf_data"
TAR_FILE="$BASE_DIR/embryo_dataset.tar.gz"
EXTRACT_DIR="$BASE_DIR/embryo_dataset"

# Check tar.gz
if [ ! -f "$TAR_FILE" ]; then
    echo "❌ Error: Tar.gz not found: $TAR_FILE"
    exit 1
fi

TAR_SIZE=$(du -sh "$TAR_FILE" | cut -f1)
echo "Tar.gz file: $TAR_FILE"
echo "Size: $TAR_SIZE"
echo ""

# Check current extraction
if [ -d "$EXTRACT_DIR" ]; then
    CURRENT_SIZE=$(du -sh "$EXTRACT_DIR" | cut -f1)
    CURRENT_CELLS=$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "Current extraction:"
    echo "  Size: $CURRENT_SIZE"
    echo "  Cell directories: $CURRENT_CELLS"
    echo ""
    
    if [ "$CURRENT_CELLS" -eq 0 ]; then
        echo "⚠️  Directory is empty, will extract fresh"
    fi
fi

echo ""
read -p "Extract now? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Cancelled"
    exit 0
fi

# Extract
echo ""
echo "Extracting (this may take 10-30 minutes)..."
echo "Logging to: $BASE_DIR/extraction.log"
echo ""

cd "$BASE_DIR"

# Remove empty directory if exists
if [ -d "$EXTRACT_DIR" ]; then
    rm -rf "$EXTRACT_DIR"
fi

# Extract with progress
tar -xzvf "$TAR_FILE" 2>&1 | tee extraction.log

EXTRACT_STATUS=$?

echo ""
echo "=== Extraction Complete ==="
echo ""

if [ $EXTRACT_STATUS -eq 0 ]; then
    if [ -d "$EXTRACT_DIR" ]; then
        EXTRACTED_SIZE=$(du -sh "$EXTRACT_DIR" | cut -f1)
        EXTRACTED_SIZE_BYTES=$(du -sb "$EXTRACT_DIR" | cut -f1)
        EXTRACTED_SIZE_GB=$(echo "scale=2; $EXTRACTED_SIZE_BYTES / 1024 / 1024 / 1024" | bc)
        EXTRACTED_CELLS=$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        EXTRACTED_FILES=$(find "$EXTRACT_DIR" -type f 2>/dev/null | wc -l)
        
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
        
        # Show first few cell directories
        if [ "$EXTRACTED_CELLS" -gt 0 ]; then
            echo ""
            echo "First 5 cell directories:"
            find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | head -5 | while read dir; do
                cell_name=$(basename "$dir")
                cell_size=$(du -sh "$dir" 2>/dev/null | cut -f1)
                echo "  $cell_name: $cell_size"
            done
        fi
    else
        echo "❌ Extraction completed but directory not found"
        echo "   Check extraction.log for details"
    fi
else
    echo "❌ Extraction failed with status: $EXTRACT_STATUS"
    echo "   Check extraction.log for error details"
fi

echo ""
echo "Check extraction.log for detailed output:"
echo "  cat $BASE_DIR/extraction.log | tail -50"

