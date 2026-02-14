#!/bin/bash
# Check staging group directory sizes
# Shows the size of datasets in staging

echo "=== Checking Staging Group Directory Sizes ==="
echo ""

# Possible staging paths
STAGING_PATHS=(
    "/staging/groups/bhaskar_group/rho9/ivf_data"
    "/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset"
    "/staging/groups/bhaskar_group/ivf"
    "/staging/groups/bhaskar_group/ivf/embryo_dataset"
)

echo "Checking staging paths..."
echo ""

for path in "${STAGING_PATHS[@]}"; do
    if [ -d "$path" ] || [ -f "$path" ]; then
        echo "📁 $path"
        if [ -f "$path" ]; then
            # It's a file (tar.gz)
            SIZE=$(du -sh "$path" 2>/dev/null | cut -f1)
            SIZE_BYTES=$(du -sb "$path" 2>/dev/null | cut -f1)
            SIZE_GB=$(echo "scale=2; $SIZE_BYTES / 1024 / 1024 / 1024" | bc 2>/dev/null)
            echo "   Type: File (tar.gz)"
            echo "   Size: $SIZE (${SIZE_GB} GB)"
        else
            # It's a directory
            SIZE=$(du -sh "$path" 2>/dev/null | cut -f1)
            SIZE_BYTES=$(du -sb "$path" 2>/dev/null | cut -f1)
            SIZE_GB=$(echo "scale=2; $SIZE_BYTES / 1024 / 1024 / 1024" | bc 2>/dev/null)
            echo "   Type: Directory"
            echo "   Size: $SIZE (${SIZE_GB} GB)"
            
            # Count subdirectories if it's a dataset directory
            if [[ "$path" == *"embryo_dataset"* ]]; then
                CELL_COUNT=$(find "$path" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
                FILE_COUNT=$(find "$path" -type f \( -name "*.jpeg" -o -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
                echo "   Cell directories: $CELL_COUNT"
                echo "   Image files: $FILE_COUNT"
            fi
        fi
        echo ""
    else
        echo "❌ $path (not found)"
        echo ""
    fi
done

# Check tar.gz file specifically
echo "=== Checking tar.gz files ==="
echo ""

TAR_FILES=(
    "/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"
    "/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz"
)

for tar_file in "${TAR_FILES[@]}"; do
    if [ -f "$tar_file" ]; then
        echo "📦 $tar_file"
        SIZE=$(du -sh "$tar_file" 2>/dev/null | cut -f1)
        SIZE_BYTES=$(du -sb "$tar_file" 2>/dev/null | cut -f1)
        SIZE_GB=$(echo "scale=2; $SIZE_BYTES / 1024 / 1024 / 1024" | bc 2>/dev/null)
        echo "   Compressed size: $SIZE (${SIZE_GB} GB)"
        
        # Try to get uncompressed size from tar
        if command -v tar >/dev/null 2>&1; then
            echo "   Checking contents..."
            FILE_COUNT=$(tar -tzf "$tar_file" 2>/dev/null | wc -l)
            echo "   Files in archive: $FILE_COUNT"
        fi
        echo ""
    fi
done

# Summary
echo "=== Summary ==="
echo ""
echo "Expected dataset size: 12 GB (uncompressed)"
echo ""
echo "To check your extracted dataset size, run:"
echo "  du -sh ~/ivf_repo/data"
echo "  du -sh ~/ivf_repo/data_raw/embryo_dataset"

