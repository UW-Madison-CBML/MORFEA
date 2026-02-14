#!/bin/bash
# Remove RI382-2 folder and check directory size

echo "=== Remove RI382-2 and Check Directory ==="
echo ""

BASE_DIR="/staging/groups/bhaskar_group/rho9/ivf_data"
RI382_DIR="$BASE_DIR/RI382-2"
EMBRYO_DIR="$BASE_DIR/embryo_dataset"

echo "1. Checking current state..."
echo ""

# Check RI382-2 folder
if [ -d "$RI382_DIR" ]; then
    SIZE=$(du -sh "$RI382_DIR" 2>/dev/null | cut -f1)
    FILE_COUNT=$(find "$RI382_DIR" -type f 2>/dev/null | wc -l)
    echo "   Found RI382-2 folder:"
    echo "     Path: $RI382_DIR"
    echo "     Size: $SIZE"
    echo "     Files: $FILE_COUNT"
    echo ""
    
    read -p "   Remove RI382-2 folder? (y/n): " confirm
    if [ "$confirm" = "y" ]; then
        rm -rf "$RI382_DIR"
        echo "   ✅ Removed: $RI382_DIR"
    else
        echo "   Skipped"
    fi
else
    echo "   ❌ RI382-2 folder not found: $RI382_DIR"
fi

echo ""

# Check embryo_dataset directory
if [ -d "$EMBRYO_DIR" ]; then
    SIZE=$(du -sh "$EMBRYO_DIR" 2>/dev/null | cut -f1)
    SIZE_BYTES=$(du -sb "$EMBRYO_DIR" 2>/dev/null | cut -f1)
    SIZE_GB=$(echo "scale=2; $SIZE_BYTES / 1024 / 1024 / 1024" | bc)
    CELL_COUNT=$(find "$EMBRYO_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    FILE_COUNT=$(find "$EMBRYO_DIR" -type f 2>/dev/null | wc -l)
    
    echo "2. embryo_dataset directory:"
    echo "   Path: $EMBRYO_DIR"
    echo "   Size: $SIZE (${SIZE_GB} GB)"
    echo "   Cell directories: $CELL_COUNT"
    echo "   Files: $FILE_COUNT"
    echo ""
    
    if [ "$CELL_COUNT" -gt 0 ]; then
        echo "   Cell directories (first 10):"
        find "$EMBRYO_DIR" -mindepth 1 -maxdepth 1 -type d | head -10 | while read dir; do
            cell_name=$(basename "$dir")
            cell_size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            cell_files=$(find "$dir" -type f 2>/dev/null | wc -l)
            echo "     $cell_name: $cell_size, $cell_files files"
        done
    fi
else
    echo "2. ❌ embryo_dataset directory not found: $EMBRYO_DIR"
fi

echo ""

# Check if RI382-2 is inside embryo_dataset
if [ -d "$EMBRYO_DIR/RI382-2" ]; then
    SIZE=$(du -sh "$EMBRYO_DIR/RI382-2" 2>/dev/null | cut -f1)
    FILE_COUNT=$(find "$EMBRYO_DIR/RI382-2" -type f 2>/dev/null | wc -l)
    echo "3. Found RI382-2 inside embryo_dataset:"
    echo "   Path: $EMBRYO_DIR/RI382-2"
    echo "   Size: $SIZE"
    echo "   Files: $FILE_COUNT"
    echo ""
    
    read -p "   Remove RI382-2 from inside embryo_dataset? (y/n): " confirm
    if [ "$confirm" = "y" ]; then
        rm -rf "$EMBRYO_DIR/RI382-2"
        echo "   ✅ Removed: $EMBRYO_DIR/RI382-2"
        
        # Re-check embryo_dataset size
        echo ""
        echo "   Updated embryo_dataset:"
        NEW_SIZE=$(du -sh "$EMBRYO_DIR" 2>/dev/null | cut -f1)
        NEW_SIZE_BYTES=$(du -sb "$EMBRYO_DIR" 2>/dev/null | cut -f1)
        NEW_SIZE_GB=$(echo "scale=2; $NEW_SIZE_BYTES / 1024 / 1024 / 1024" | bc)
        NEW_CELL_COUNT=$(find "$EMBRYO_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        echo "     Size: $NEW_SIZE (${NEW_SIZE_GB} GB)"
        echo "     Cell directories: $NEW_CELL_COUNT"
    else
        echo "   Skipped"
    fi
else
    echo "3. RI382-2 not found inside embryo_dataset"
fi

echo ""
echo "=== Summary ==="
echo ""

# Final check of base directory
if [ -d "$BASE_DIR" ]; then
    echo "Base directory: $BASE_DIR"
    echo "Contents:"
    ls -lh "$BASE_DIR" | grep -E "^d|^-" | head -10
    echo ""
    
    TOTAL_SIZE=$(du -sh "$BASE_DIR" 2>/dev/null | cut -f1)
    echo "Total size of $BASE_DIR: $TOTAL_SIZE"
fi

