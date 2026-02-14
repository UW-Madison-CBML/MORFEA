#!/bin/bash
# Find where Jens's dataset actually is

echo "=== Searching for Jens's Dataset ==="
echo ""

# Possible locations
SEARCH_PATHS=(
    "/staging/groups/bhaskar_group/ivf"
    "/staging/groups/bhaskar_group/ivf/embryo_dataset"
    "/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz"
    "/staging/groups/bhaskar_group/jlundsgaard"
    "/staging/groups/bhaskar_group/jlundsgaard/ivf"
    "/staging/groups/bhaskar_group/jlundsgaard/embryo_dataset"
    "/project/bhaskar_group/ivf"
    "/project/bhaskar_group/ivf/embryo_dataset"
    "/home/jlundsgaard"
    "/home/jlundsgaard/ivf"
)

echo "1. Checking common paths..."
echo ""

FOUND=false
for path in "${SEARCH_PATHS[@]}"; do
    if [ -e "$path" ]; then
        echo "✅ Found: $path"
        if [ -d "$path" ]; then
            SIZE=$(du -sh "$path" 2>/dev/null | cut -f1)
            echo "   Type: Directory"
            echo "   Size: $SIZE"
            if [[ "$path" == *"embryo_dataset"* ]]; then
                CELL_COUNT=$(find "$path" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
                echo "   Cell directories: $CELL_COUNT"
                FOUND=true
            fi
        elif [ -f "$path" ]; then
            SIZE=$(du -sh "$path" 2>/dev/null | cut -f1)
            echo "   Type: File"
            echo "   Size: $SIZE"
            if [[ "$path" == *".tar.gz"* ]]; then
                FOUND=true
            fi
        fi
        echo ""
    fi
done

if [ "$FOUND" = false ]; then
    echo "❌ Dataset not found in common paths"
    echo ""
fi

echo "2. Searching /staging/groups/bhaskar_group/ for embryo-related files..."
echo ""

# Search for embryo_dataset in bhaskar_group
find /staging/groups/bhaskar_group/ -maxdepth 3 -name "*embryo*" -o -name "*ivf*" 2>/dev/null | head -20 | while read item; do
    if [ -e "$item" ]; then
        if [ -d "$item" ]; then
            SIZE=$(du -sh "$item" 2>/dev/null | cut -f1)
            echo "  Directory: $item ($SIZE)"
        elif [ -f "$item" ]; then
            SIZE=$(du -sh "$item" 2>/dev/null | cut -f1)
            echo "  File: $item ($SIZE)"
        fi
    fi
done

echo ""
echo "3. Checking what's in /staging/groups/bhaskar_group/ivf/ (if exists)..."
if [ -d "/staging/groups/bhaskar_group/ivf" ]; then
    echo "   Contents:"
    ls -lh /staging/groups/bhaskar_group/ivf/ | head -10
else
    echo "   ❌ /staging/groups/bhaskar_group/ivf/ does not exist"
fi

echo ""
echo "4. Your current dataset:"
YOUR_PATH="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset"
if [ -d "$YOUR_PATH" ]; then
    SIZE=$(du -sh "$YOUR_PATH" 2>/dev/null | cut -f1)
    CELL_COUNT=$(find "$YOUR_PATH" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "   ✅ Found at: $YOUR_PATH"
    echo "   Size: $SIZE"
    echo "   Cell directories: $CELL_COUNT"
else
    echo "   ❌ Not found at: $YOUR_PATH"
fi

YOUR_TAR="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"
if [ -f "$YOUR_TAR" ]; then
    SIZE=$(du -sh "$YOUR_TAR" 2>/dev/null | cut -f1)
    echo "   ✅ Tar.gz found at: $YOUR_TAR"
    echo "   Size: $SIZE"
fi

echo ""
echo "=== Conclusion ==="
echo ""
if [ "$FOUND" = false ]; then
    echo "Jens's dataset not found. Possible reasons:"
    echo "  1. Dataset was moved or deleted"
    echo "  2. Different path than expected"
    echo "  3. Dataset is in a different location"
    echo ""
    echo "Recommendation:"
    echo "  - Use your own dataset at /staging/groups/bhaskar_group/rho9/ivf_data/"
    echo "  - Or ask Jens for the correct path"
else
    echo "Found potential dataset location(s) above"
fi

