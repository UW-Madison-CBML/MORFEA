#!/bin/bash
# Check where the actual embryo data is located

echo "=== Checking Data Locations ==="
echo ""

# Check if embryo_dataset is empty
EMBRYO_DIR="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset"
if [ -d "$EMBRYO_DIR" ]; then
    COUNT=$(ls "$EMBRYO_DIR" 2>/dev/null | wc -l)
    echo "1. $EMBRYO_DIR:"
    echo "   Cell directories: $COUNT"
    if [ "$COUNT" -eq 0 ]; then
        echo "   ⚠️  Directory is EMPTY"
    else
        echo "   ✓ Has data"
        echo "   First 10 cells:"
        ls "$EMBRYO_DIR" | head -10
    fi
fi

echo ""
echo "2. Checking tar.gz file:"
TAR_FILE="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"
if [ -f "$TAR_FILE" ]; then
    SIZE=$(du -sh "$TAR_FILE" | cut -f1)
    echo "   ✓ Found: $TAR_FILE ($SIZE)"
    echo "   Need to extract? Run:"
    echo "     cd /staging/groups/bhaskar_group/rho9/ivf_data"
    echo "     tar -xzvf embryo_dataset.tar.gz"
else
    echo "   ✗ Not found: $TAR_FILE"
fi

echo ""
echo "3. Checking alternative locations:"
ALT_PATHS=(
    "/staging/groups/bhaskar_group/ivf/embryo_dataset"
    "/staging/groups/bhaskar_group/rho9/ivf_data"
    "~/ivf_repo/data"
)

for path in "${ALT_PATHS[@]}"; do
    expanded_path=$(eval echo "$path")
    if [ -d "$expanded_path" ]; then
        COUNT=$(ls "$expanded_path" 2>/dev/null | wc -l)
        if [ "$COUNT" -gt 0 ]; then
            echo "   ✓ $expanded_path: $COUNT items"
            if [ "$COUNT" -lt 100 ]; then
                echo "      First 5: $(ls "$expanded_path" | head -5 | tr '\n' ' ')"
            fi
        fi
    fi
done

echo ""
echo "4. Recommendation:"
if [ -f "$TAR_FILE" ] && [ "$COUNT" -eq 0 ]; then
    echo "   Extract the tar.gz file:"
    echo "     cd /staging/groups/bhaskar_group/rho9/ivf_data"
    echo "     tar -xzvf embryo_dataset.tar.gz"
    echo "     # This will take 10-30 minutes"
fi

