#!/bin/bash
# Check Jens's ivf dataset size and structure

echo "=== Jens's IVF Dataset Analysis ==="
echo ""

JENS_BASE="/staging/groups/bhaskar_group/ivf"

echo "1. Total directory size:"
TOTAL_SIZE=$(du -sh "$JENS_BASE" 2>/dev/null | cut -f1)
TOTAL_SIZE_BYTES=$(du -sb "$JENS_BASE" 2>/dev/null | cut -f1)
TOTAL_SIZE_GB=$(echo "scale=2; $TOTAL_SIZE_BYTES / 1024 / 1024 / 1024" | bc)
echo "   $JENS_BASE: $TOTAL_SIZE (${TOTAL_SIZE_GB} GB)"
echo ""

echo "2. Directory contents:"
ls -lh "$JENS_BASE"
echo ""

echo "3. Breakdown by file/directory:"
echo ""
for item in "$JENS_BASE"/*; do
    if [ -e "$item" ]; then
        item_name=$(basename "$item")
        if [ -d "$item" ]; then
            SIZE=$(du -sh "$item" 2>/dev/null | cut -f1)
            SIZE_BYTES=$(du -sb "$item" 2>/dev/null | cut -f1)
            SIZE_GB=$(echo "scale=2; $SIZE_BYTES / 1024 / 1024 / 1024" | bc)
            echo "   📁 $item_name: $SIZE (${SIZE_GB} GB)"
            
            # If it's embryo_dataset directory, check cell count
            if [ "$item_name" = "embryo_dataset" ]; then
                CELL_COUNT=$(find "$item" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
                FILE_COUNT=$(find "$item" -type f 2>/dev/null | wc -l)
                echo "      Cell directories: $CELL_COUNT"
                echo "      Files: $FILE_COUNT"
            fi
        elif [ -f "$item" ]; then
            SIZE=$(du -sh "$item" 2>/dev/null | cut -f1)
            SIZE_BYTES=$(du -sb "$item" 2>/dev/null | cut -f1)
            SIZE_GB=$(echo "scale=2; $SIZE_BYTES / 1024 / 1024 / 1024" | bc)
            echo "   📄 $item_name: $SIZE (${SIZE_GB} GB)"
            
            # If it's tar.gz, check contents
            if [[ "$item_name" == *.tar.gz ]]; then
                echo "      Type: tar.gz archive"
                if tar -tzf "$item" >/dev/null 2>&1; then
                    FILE_COUNT=$(tar -tzf "$item" 2>/dev/null | wc -l)
                    echo "      Files inside: $FILE_COUNT"
                    if [[ "$item_name" == *"annotations"* ]]; then
                        echo "      ⚠️  This is annotations/metadata only, not image dataset"
                    fi
                fi
            fi
        fi
    fi
done

echo ""
echo "4. Looking for actual image dataset:"
echo ""

JENS_DATASET_TAR="$JENS_BASE/embryo_dataset.tar.gz"
JENS_DATASET_DIR="$JENS_BASE/embryo_dataset"

if [ -f "$JENS_DATASET_TAR" ]; then
    SIZE=$(du -sh "$JENS_DATASET_TAR" | cut -f1)
    SIZE_BYTES=$(du -sb "$JENS_DATASET_TAR" | cut -f1)
    SIZE_GB=$(echo "scale=2; $SIZE_BYTES / 1024 / 1024 / 1024" | bc)
    echo "   ✅ Found image dataset tar.gz:"
    echo "      $JENS_DATASET_TAR"
    echo "      Size: $SIZE (${SIZE_GB} GB)"
    echo "      Readable: $([ -r "$JENS_DATASET_TAR" ] && echo "YES" || echo "NO")"
    IMAGE_DATASET_SIZE="$SIZE_GB"
elif [ -d "$JENS_DATASET_DIR" ]; then
    SIZE=$(du -sh "$JENS_DATASET_DIR" 2>/dev/null | cut -f1)
    SIZE_BYTES=$(du -sb "$JENS_DATASET_DIR" 2>/dev/null | cut -f1)
    SIZE_GB=$(echo "scale=2; $SIZE_BYTES / 1024 / 1024 / 1024" | bc)
    CELL_COUNT=$(find "$JENS_DATASET_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    FILE_COUNT=$(find "$JENS_DATASET_DIR" -type f 2>/dev/null | wc -l)
    echo "   ✅ Found extracted image dataset:"
    echo "      $JENS_DATASET_DIR"
    echo "      Size: $SIZE (${SIZE_GB} GB)"
    echo "      Cell directories: $CELL_COUNT"
    echo "      Files: $FILE_COUNT"
    echo "      Readable: $([ -r "$JENS_DATASET_DIR" ] && echo "YES" || echo "NO")"
    IMAGE_DATASET_SIZE="$SIZE_GB"
else
    echo "   ❌ No image dataset found (only annotations/metadata)"
    echo "      The 2.9G might be other files, not the actual image dataset"
    IMAGE_DATASET_SIZE="0"
fi

echo ""
echo "5. Comparison:"
echo ""
echo "   Your dataset: 12 GB (tar.gz)"
if [ -n "$IMAGE_DATASET_SIZE" ] && [ "$IMAGE_DATASET_SIZE" != "0" ]; then
    echo "   Jens's dataset: ${IMAGE_DATASET_SIZE} GB"
    DIFF=$(echo "scale=2; 12 - $IMAGE_DATASET_SIZE" | bc)
    if (( $(echo "$DIFF > 0" | bc -l) )); then
        echo "   Difference: Your dataset is ${DIFF} GB larger"
    else
        DIFF_ABS=$(echo "scale=2; $IMAGE_DATASET_SIZE - 12" | bc)
        echo "   Difference: Jens's dataset is ${DIFF_ABS} GB larger"
    fi
else
    echo "   Jens's dataset: Not found (only 2.9G total, likely other files)"
fi

echo ""
echo "6. Summary:"
echo "   Total in /staging/groups/bhaskar_group/ivf/: ${TOTAL_SIZE_GB} GB"
echo "   This includes:"
echo "     - Annotations: 53K (metadata only)"
echo "     - Grades CSV: small"
if [ -n "$IMAGE_DATASET_SIZE" ] && [ "$IMAGE_DATASET_SIZE" != "0" ]; then
    echo "     - Image dataset: ${IMAGE_DATASET_SIZE} GB"
else
    echo "     - Image dataset: NOT FOUND"
    echo ""
    echo "   ⚠️  The actual image dataset might be:"
    echo "      - In a different location"
    echo "      - Not yet uploaded to staging"
    echo "      - Ask Jens for the correct path"
fi

