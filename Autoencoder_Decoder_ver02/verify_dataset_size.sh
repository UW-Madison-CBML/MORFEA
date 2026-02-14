#!/bin/bash
# Verify dataset size and check for missing files
# Compares extracted dataset size with expected size

echo "=== Dataset Size Verification ==="
echo ""

# Check if data directory exists
if [ -L "data" ]; then
    DATA_DIR=$(readlink -f data)
elif [ -d "data" ]; then
    DATA_DIR="data"
else
    echo "❌ Error: 'data' directory not found"
    echo "   Please run extraction first or create a symlink"
    exit 1
fi

echo "Data directory: $DATA_DIR"
echo ""

# Calculate actual size
echo "Calculating dataset size..."
ACTUAL_SIZE=$(du -sb "$DATA_DIR" 2>/dev/null | cut -f1)
ACTUAL_SIZE_GB=$(echo "scale=2; $ACTUAL_SIZE / 1024 / 1024 / 1024" | bc)

echo "Actual size: ${ACTUAL_SIZE_GB} GB"
echo "Expected size: 12 GB"
echo ""

# Calculate difference
DIFF_GB=$(echo "scale=2; 12 - $ACTUAL_SIZE_GB" | bc)
DIFF_PERCENT=$(echo "scale=1; ($DIFF_GB / 12) * 100" | bc)

echo "Difference: ${DIFF_GB} GB (${DIFF_PERCENT}%)"
echo ""

# Check if size is significantly different
if (( $(echo "$ACTUAL_SIZE_GB < 11.5" | bc -l) )); then
    echo "⚠️  WARNING: Dataset size is significantly smaller than expected!"
    echo "   This may indicate missing files or incomplete extraction."
    echo ""
fi

# Count cell directories
echo "Counting cell directories..."
CELL_COUNT=$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Cell directories found: $CELL_COUNT"
echo ""

# Count total image files
echo "Counting image files..."
IMAGE_COUNT=$(find "$DATA_DIR" -type f \( -name "*.jpeg" -o -name "*.jpg" -o -name "*.png" \) | wc -l)
echo "Total image files: $IMAGE_COUNT"
echo ""

# Calculate average size per cell
if [ "$CELL_COUNT" -gt 0 ]; then
    AVG_SIZE_PER_CELL=$(echo "scale=2; $ACTUAL_SIZE_GB / $CELL_COUNT" | bc)
    echo "Average size per cell: ${AVG_SIZE_PER_CELL} GB"
    echo ""
fi

# Check for empty directories
echo "Checking for empty cell directories..."
EMPTY_DIRS=$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -empty | wc -l)
if [ "$EMPTY_DIRS" -gt 0 ]; then
    echo "⚠️  Found $EMPTY_DIRS empty cell directories:"
    find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -empty | head -5
    echo ""
fi

# Check tar.gz file size if it exists
STAGING_TAR="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"
if [ -f "$STAGING_TAR" ]; then
    echo "Checking original tar.gz file..."
    TAR_SIZE=$(du -sb "$STAGING_TAR" 2>/dev/null | cut -f1)
    TAR_SIZE_GB=$(echo "scale=2; $TAR_SIZE / 1024 / 1024 / 1024" | bc)
    echo "Tar.gz size: ${TAR_SIZE_GB} GB"
    echo ""
    
    # Compare compressed vs extracted
    COMPRESSION_RATIO=$(echo "scale=2; $TAR_SIZE_GB / $ACTUAL_SIZE_GB" | bc)
    echo "Compression ratio: ${COMPRESSION_RATIO}x"
    echo ""
fi

# Sample a few cell directories to check file counts
echo "Sampling cell directories (first 5)..."
SAMPLE_COUNT=0
for cell_dir in $(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | head -5); do
    cell_name=$(basename "$cell_dir")
    file_count=$(find "$cell_dir" -type f \( -name "*.jpeg" -o -name "*.jpg" -o -name "*.png" \) | wc -l)
    dir_size=$(du -sh "$cell_dir" 2>/dev/null | cut -f1)
    echo "  $cell_name: $file_count files, $dir_size"
    SAMPLE_COUNT=$((SAMPLE_COUNT + file_count))
done
echo ""

# Summary
echo "=== Summary ==="
echo "Extracted size: ${ACTUAL_SIZE_GB} GB (expected: 12 GB)"
echo "Missing: ${DIFF_GB} GB (${DIFF_PERCENT}%)"
echo "Cell directories: $CELL_COUNT"
echo "Total images: $IMAGE_COUNT"
echo ""

if (( $(echo "$ACTUAL_SIZE_GB < 11.5" | bc -l) )); then
    echo "⚠️  RECOMMENDATION:"
    echo "   1. Check if tar.gz extraction completed without errors"
    echo "   2. Verify tar.gz file integrity: tar -tzf <tar_file> | wc -l"
    echo "   3. Re-extract if necessary"
    echo "   4. Check disk space during extraction"
else
    echo "✅ Dataset size looks reasonable (within 5% of expected)"
fi

