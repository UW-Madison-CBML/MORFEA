#!/bin/bash
# Diagnose why extraction only got 0.08 GB instead of 12 GB

echo "=== Diagnosing Extraction Issue ==="
echo ""

TAR_FILE="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"
EXTRACTED_DIR="/mnt/htc-cephfs/fuse/root/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset"

echo "1. Checking tar.gz file..."
if [ -f "$TAR_FILE" ]; then
    TAR_SIZE=$(du -sh "$TAR_FILE" | cut -f1)
    TAR_SIZE_BYTES=$(du -sb "$TAR_FILE" | cut -f1)
    echo "   Size: $TAR_SIZE"
    echo "   Checking contents..."
    
    # Count files in tar.gz
    echo "   Counting files in tar.gz (this may take a while)..."
    TAR_FILE_COUNT=$(tar -tzf "$TAR_FILE" 2>/dev/null | wc -l)
    echo "   Files in tar.gz: $TAR_FILE_COUNT"
    
    # Count cell directories in tar.gz
    TAR_CELL_COUNT=$(tar -tzf "$TAR_FILE" 2>/dev/null | grep -E '^[^/]+/$' | wc -l)
    echo "   Cell directories in tar.gz: $TAR_CELL_COUNT"
    
    # Show first few cell directories
    echo "   First 10 cell directories in tar.gz:"
    tar -tzf "$TAR_FILE" 2>/dev/null | grep -E '^[^/]+/$' | head -10 | sed 's/^/     /'
else
    echo "   ❌ Tar.gz file not found!"
    exit 1
fi

echo ""
echo "2. Checking extracted directory..."
if [ -d "$EXTRACTED_DIR" ]; then
    EXTRACTED_SIZE=$(du -sh "$EXTRACTED_DIR" | cut -f1)
    EXTRACTED_CELL_COUNT=$(find "$EXTRACTED_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
    EXTRACTED_FILE_COUNT=$(find "$EXTRACTED_DIR" -type f | wc -l)
    echo "   Size: $EXTRACTED_SIZE"
    echo "   Cell directories: $EXTRACTED_CELL_COUNT"
    echo "   Files: $EXTRACTED_FILE_COUNT"
    
    echo "   Cell directories found:"
    find "$EXTRACTED_DIR" -mindepth 1 -maxdepth 1 -type d | head -10 | sed 's/^/     /'
else
    echo "   ❌ Extracted directory not found!"
    exit 1
fi

echo ""
echo "3. Comparison:"
echo "   Tar.gz contains: $TAR_FILE_COUNT files, $TAR_CELL_COUNT cell directories"
echo "   Extracted has: $EXTRACTED_FILE_COUNT files, $EXTRACTED_CELL_COUNT cell directories"
echo ""

MISSING_CELLS=$((TAR_CELL_COUNT - EXTRACTED_CELL_COUNT))
MISSING_FILES=$((TAR_FILE_COUNT - EXTRACTED_FILE_COUNT))

if [ "$MISSING_CELLS" -gt 0 ] || [ "$MISSING_FILES" -gt 0 ]; then
    echo "⚠️  PROBLEM DETECTED:"
    echo "   Missing cell directories: $MISSING_CELLS"
    echo "   Missing files: $MISSING_FILES"
    echo ""
    echo "🔧 SOLUTION: Re-extract the tar.gz file"
    echo ""
    echo "   Steps to re-extract:"
    echo "   1. Remove old extraction:"
    echo "      rm -rf $EXTRACTED_DIR"
    echo ""
    echo "   2. Create new directory:"
    echo "      mkdir -p $EXTRACTED_DIR"
    echo ""
    echo "   3. Re-extract (with verbose output to see errors):"
    echo "      tar -xzvf $TAR_FILE -C $(dirname $EXTRACTED_DIR) 2>&1 | tee extraction.log"
    echo ""
    echo "   4. Check for errors in extraction.log"
    echo ""
    echo "   OR extract to a different location first:"
    echo "      mkdir -p ~/ivf_repo/data_raw"
    echo "      cd ~/ivf_repo/data_raw"
    echo "      tar -xzvf $TAR_FILE 2>&1 | tee extraction.log"
else
    echo "✅ File counts match, but size is still wrong"
    echo "   This might indicate:"
    echo "   - Corrupted files in tar.gz"
    echo "   - Files extracted but then deleted"
    echo "   - Symbolic links pointing to wrong location"
fi

