#!/bin/bash
# Check staging capacity and Jens's dataset accessibility

echo "=== Staging Capacity Check ==="
echo ""

# Check disk space
echo "1. Disk space on /staging:"
df -h /staging 2>/dev/null || df -h | grep staging
echo ""

# Check bhaskar_group usage
echo "2. bhaskar_group directory usage:"
if [ -d "/staging/groups/bhaskar_group" ]; then
    TOTAL_SIZE=$(du -sh /staging/groups/bhaskar_group 2>/dev/null | cut -f1)
    echo "   Total size: $TOTAL_SIZE"
    echo ""
    
    echo "   Breakdown by subdirectory:"
    for dir in /staging/groups/bhaskar_group/*; do
        if [ -d "$dir" ]; then
            dir_name=$(basename "$dir")
            dir_size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo "     $dir_name: $dir_size"
        fi
    done
fi

echo ""
echo "=== Jens's Dataset Check ==="
echo ""

JENS_BASE="/staging/groups/bhaskar_group/ivf"
JENS_ANNOTATIONS="$JENS_BASE/embryo_dataset_annotations.tar.gz"
JENS_CSV="$JENS_BASE/embryo_dataset_grades.csv"

echo "3. Jens's directory: $JENS_BASE"
if [ -d "$JENS_BASE" ]; then
    echo "   ✅ Directory exists"
    echo "   Contents:"
    ls -lh "$JENS_BASE"
    echo ""
    
    TOTAL_SIZE=$(du -sh "$JENS_BASE" 2>/dev/null | cut -f1)
    echo "   Total size: $TOTAL_SIZE"
else
    echo "   ❌ Directory not found"
fi

echo ""
echo "4. Checking embryo_dataset_annotations.tar.gz:"
if [ -f "$JENS_ANNOTATIONS" ]; then
    echo "   ✅ File exists: $JENS_ANNOTATIONS"
    
    # File info
    SIZE=$(du -sh "$JENS_ANNOTATIONS" | cut -f1)
    SIZE_BYTES=$(du -sb "$JENS_ANNOTATIONS" | cut -f1)
    SIZE_GB=$(echo "scale=2; $SIZE_BYTES / 1024 / 1024 / 1024" | bc)
    
    ls -lh "$JENS_ANNOTATIONS"
    echo ""
    echo "   Size: $SIZE (${SIZE_GB} GB)"
    echo "   Owner: $(stat -c '%U' "$JENS_ANNOTATIONS" 2>/dev/null || stat -f '%Su' "$JENS_ANNOTATIONS" 2>/dev/null)"
    echo "   Group: $(stat -c '%G' "$JENS_ANNOTATIONS" 2>/dev/null || stat -f '%Sg' "$JENS_ANNOTATIONS" 2>/dev/null)"
    echo "   Permissions: $(stat -c '%a' "$JENS_ANNOTATIONS" 2>/dev/null || stat -f '%A' "$JENS_ANNOTATIONS" 2>/dev/null)"
    echo ""
    
    # Check read permission
    if [ -r "$JENS_ANNOTATIONS" ]; then
        echo "   ✅ Read permission: YES"
    else
        echo "   ❌ Read permission: NO"
    fi
    
    # Check if it's a valid tar.gz
    echo ""
    echo "   Checking if valid tar.gz..."
    if tar -tzf "$JENS_ANNOTATIONS" >/dev/null 2>&1; then
        echo "   ✅ Valid tar.gz file"
        echo ""
        echo "   Contents (first 20 files):"
        tar -tzf "$JENS_ANNOTATIONS" 2>/dev/null | head -20
        echo ""
        FILE_COUNT=$(tar -tzf "$JENS_ANNOTATIONS" 2>/dev/null | wc -l)
        echo "   Total files in archive: $FILE_COUNT"
    else
        echo "   ❌ Not a valid tar.gz or corrupted"
    fi
else
    echo "   ❌ File not found: $JENS_ANNOTATIONS"
fi

echo ""
echo "5. Checking for actual embryo_dataset (not just annotations):"
echo ""

# Look for actual dataset files
JENS_DATASET_TAR="$JENS_BASE/embryo_dataset.tar.gz"
JENS_DATASET_DIR="$JENS_BASE/embryo_dataset"

if [ -f "$JENS_DATASET_TAR" ]; then
    SIZE=$(du -sh "$JENS_DATASET_TAR" | cut -f1)
    echo "   ✅ Found: $JENS_DATASET_TAR ($SIZE)"
    if [ -r "$JENS_DATASET_TAR" ]; then
        echo "      Readable: YES"
    else
        echo "      Readable: NO"
    fi
fi

if [ -d "$JENS_DATASET_DIR" ]; then
    SIZE=$(du -sh "$JENS_DATASET_DIR" 2>/dev/null | cut -f1)
    CELL_COUNT=$(find "$JENS_DATASET_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "   ✅ Found: $JENS_DATASET_DIR ($SIZE, $CELL_COUNT cells)"
    if [ -r "$JENS_DATASET_DIR" ]; then
        echo "      Readable: YES"
    else
        echo "      Readable: NO"
    fi
fi

if [ ! -f "$JENS_DATASET_TAR" ] && [ ! -d "$JENS_DATASET_DIR" ]; then
    echo "   ⚠️  No actual embryo_dataset found (only annotations)"
    echo "      The annotations.tar.gz is likely just metadata, not the image dataset"
fi

echo ""
echo "6. Summary:"
echo ""

# Your dataset
if [ -d "/staging/groups/bhaskar_group/rho9/ivf_data" ]; then
    YOUR_SIZE=$(du -sh /staging/groups/bhaskar_group/rho9/ivf_data 2>/dev/null | cut -f1)
    echo "   Your dataset: ${YOUR_SIZE}"
fi

# Jens's dataset
if [ -d "$JENS_BASE" ]; then
    JENS_SIZE=$(du -sh "$JENS_BASE" 2>/dev/null | cut -f1)
    echo "   Jens's directory: ${JENS_SIZE}"
fi

echo ""
echo "=== Recommendation ==="
echo ""

if [ -f "$JENS_ANNOTATIONS" ] && [ -r "$JENS_ANNOTATIONS" ]; then
    echo "✅ You CAN access Jens's annotations file"
    echo "   However, this appears to be annotations/metadata only"
    echo "   You may need the actual image dataset (embryo_dataset.tar.gz)"
    echo ""
    echo "   To use annotations:"
    echo "     tar -xzf $JENS_ANNOTATIONS"
else
    echo "❌ Cannot access Jens's annotations file"
fi

if [ ! -f "$JENS_DATASET_TAR" ] && [ ! -d "$JENS_DATASET_DIR" ]; then
    echo ""
    echo "⚠️  Note: Only annotations found, not the actual image dataset"
    echo "   You may need to ask Jens for the image dataset location"
fi

