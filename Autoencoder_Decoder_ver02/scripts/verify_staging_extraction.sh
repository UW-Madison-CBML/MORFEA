#!/bin/bash
# Verify if the staging extraction is complete

set -e

echo "============================================================"
echo "Verifying Staging Extraction"
echo "============================================================"
echo ""

IVF_DATA_DIR="/staging/groups/bhaskar_group/rho9/ivf_data"
EXTRACT_DIR="$IVF_DATA_DIR/embryo_dataset"

# Check if extracted directory exists
if [ -d "$EXTRACT_DIR" ]; then
    echo "✓ Found extracted dataset directory"
    echo ""
    
    # Get size
    EXTRACTED_SIZE=$(du -sh "$EXTRACT_DIR" 2>/dev/null | cut -f1)
    echo "Size: $EXTRACTED_SIZE"
    
    # Count cell directories
    CELL_COUNT=$(ls -d "$EXTRACT_DIR"/*/ 2>/dev/null | wc -l)
    echo "Cell directories: $CELL_COUNT"
    echo ""
    
    # Check if it looks complete
    if [ "$CELL_COUNT" -gt 100 ]; then
        echo "✅ Looks COMPLETE! (Many cell directories found)"
    elif [ "$CELL_COUNT" -gt 10 ]; then
        echo "⚠️  Partial extraction? (Some cells found, but might be incomplete)"
    else
        echo "❌ Looks INCOMPLETE! (Very few cells)"
    fi
    echo ""
    
    # Show first 10 cells
    echo "First 10 cell directories:"
    ls -d "$EXTRACT_DIR"/*/ 2>/dev/null | head -10 | xargs -n1 basename
    echo ""
    
    # Check a sample cell for frame count
    FIRST_CELL=$(ls -d "$EXTRACT_DIR"/*/ 2>/dev/null | head -1)
    if [ -n "$FIRST_CELL" ]; then
        FRAME_COUNT=$(ls "$FIRST_CELL"/*.jpeg 2>/dev/null | wc -l)
        CELL_NAME=$(basename "$FIRST_CELL")
        echo "Sample cell '$CELL_NAME' has $FRAME_COUNT frames"
    fi
    
else
    echo "❌ No extracted dataset directory found at:"
    echo "   $EXTRACT_DIR"
    echo ""
    echo "The tar.gz file exists but hasn't been extracted yet."
fi

echo ""
echo "============================================================"
echo "Checking for incomplete home directory extraction"
echo "============================================================"
echo ""

HOME_EXTRACT="$HOME/ivf_repo/data_raw/embryo_dataset"
if [ -d "$HOME_EXTRACT" ]; then
    HOME_SIZE=$(du -sh "$HOME_EXTRACT" 2>/dev/null | cut -f1)
    HOME_COUNT=$(ls -d "$HOME_EXTRACT"/*/ 2>/dev/null | wc -l)
    echo "⚠️  Found extraction in HOME directory:"
    echo "   Location: $HOME_EXTRACT"
    echo "   Size: $HOME_SIZE"
    echo "   Cell directories: $HOME_COUNT"
    echo ""
    echo "   → This is likely incomplete and taking up home quota!"
    echo "   → You can safely delete it if staging extraction is complete"
    echo ""
    echo "   To delete: rm -rf $HOME_EXTRACT"
else
    echo "✓ No incomplete extraction in home directory"
fi

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"

