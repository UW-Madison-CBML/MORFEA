#!/bin/bash
# Optimize storage usage - check what can be safely removed

echo "=== Storage Optimization Analysis ==="
echo ""

YOUR_BASE="/staging/groups/bhaskar_group/rho9/ivf_data"
YOUR_TAR="$YOUR_BASE/embryo_dataset.tar.gz"
YOUR_DIR="$YOUR_BASE/embryo_dataset"

echo "1. Your dataset storage:"
echo ""

if [ -f "$YOUR_TAR" ]; then
    TAR_SIZE=$(du -sb "$YOUR_TAR" | cut -f1)
    TAR_SIZE_GB=$(echo "scale=2; $TAR_SIZE / 1024 / 1024 / 1024" | bc)
    echo "   Tar.gz: ${TAR_SIZE_GB} GB"
fi

if [ -d "$YOUR_DIR" ]; then
    DIR_SIZE=$(du -sb "$YOUR_DIR" 2>/dev/null | cut -f1)
    DIR_SIZE_GB=$(echo "scale=2; $DIR_SIZE / 1024 / 1024 / 1024" | bc)
    CELL_COUNT=$(find "$YOUR_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "   Extracted directory: ${DIR_SIZE_GB} GB ($CELL_COUNT cells)"
    
    if [ "$CELL_COUNT" -eq 0 ]; then
        echo "   ⚠️  Directory is empty - safe to remove"
        CAN_REMOVE_DIR=true
    else
        echo "   ⚠️  Directory has data - removing will require re-extraction"
        CAN_REMOVE_DIR=false
    fi
else
    echo "   Extracted directory: Not found (already removed or not extracted)"
    CAN_REMOVE_DIR=false
fi

echo ""
echo "2. Storage options:"
echo ""

if [ -f "$YOUR_TAR" ] && [ -d "$YOUR_DIR" ]; then
    TOTAL_SIZE=$(du -sb "$YOUR_BASE" 2>/dev/null | cut -f1)
    TOTAL_SIZE_GB=$(echo "scale=2; $TOTAL_SIZE / 1024 / 1024 / 1024" | bc)
    echo "   Current total: ${TOTAL_SIZE_GB} GB"
    echo ""
    echo "   Option A: Keep both tar.gz and extracted (${TOTAL_SIZE_GB} GB)"
    echo "      - Fast access to data"
    echo "      - No need to extract when needed"
    echo ""
    echo "   Option B: Keep only tar.gz, remove extracted (${TAR_SIZE_GB} GB)"
    echo "      - Saves ~${DIR_SIZE_GB} GB"
    echo "      - Need to extract when needed (10-30 min)"
    echo ""
    echo "   Option C: Remove everything"
    echo "      - Saves all ${TOTAL_SIZE_GB} GB"
    echo "      - Need to re-upload if needed later"
fi

echo ""
echo "3. Context:"
echo "   Available staging space: ~2000 TB (2.0P)"
echo "   Your dataset: ~12 GB"
echo "   Percentage of available space: ~0.0006%"
echo ""
echo "   💡 Recommendation: Keep your dataset"
echo "      - 12 GB is negligible compared to available space"
echo "      - You need it for your analysis"
echo "      - Re-extraction takes time"

echo ""
echo "4. If you still want to optimize:"
echo ""

if [ "$CAN_REMOVE_DIR" = true ]; then
    echo "   Empty directory can be removed:"
    echo "     rm -rf $YOUR_DIR"
    echo "     (Saves minimal space since it's empty)"
elif [ -d "$YOUR_DIR" ] && [ -f "$YOUR_TAR" ]; then
    echo "   You can remove extracted directory (keep tar.gz):"
    echo "     rm -rf $YOUR_DIR"
    DIR_SIZE_GB=$(echo "scale=2; $(du -sb "$YOUR_DIR" 2>/dev/null | cut -f1) / 1024 / 1024 / 1024" | bc)
    echo "     (Saves ~${DIR_SIZE_GB} GB, but need to extract when needed)"
fi

echo ""
read -p "Remove extracted directory? (y/n): " confirm
if [ "$confirm" = "y" ] && [ -d "$YOUR_DIR" ]; then
    rm -rf "$YOUR_DIR"
    echo "✅ Removed: $YOUR_DIR"
    echo ""
    echo "Current storage:"
    du -sh "$YOUR_BASE"
else
    echo "Kept as is"
fi

