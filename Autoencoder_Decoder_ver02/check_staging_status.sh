#!/bin/bash
# Check staging directory structure and capacity, including Jens's dataset

echo "=== Staging Directory Status ==="
echo ""

# Check base staging directory
STAGING_BASE="/staging/groups/bhaskar_group"

echo "1. Base staging directory: $STAGING_BASE"
if [ -d "$STAGING_BASE" ]; then
    echo "   ✅ Exists"
    echo "   Contents:"
    ls -lh "$STAGING_BASE" | head -10
    echo ""
else
    echo "   ❌ Not found"
fi

echo ""
echo "2. Your dataset (rho9):"
echo ""

YOUR_BASE="/staging/groups/bhaskar_group/rho9/ivf_data"
YOUR_TAR="$YOUR_BASE/embryo_dataset.tar.gz"
YOUR_DIR="$YOUR_BASE/embryo_dataset"

if [ -d "$YOUR_BASE" ]; then
    echo "   Directory: $YOUR_BASE"
    TOTAL_SIZE=$(du -sh "$YOUR_BASE" 2>/dev/null | cut -f1)
    echo "   Total size: $TOTAL_SIZE"
    echo ""
    
    if [ -f "$YOUR_TAR" ]; then
        TAR_SIZE=$(du -sh "$YOUR_TAR" | cut -f1)
        TAR_SIZE_BYTES=$(du -sb "$YOUR_TAR" | cut -f1)
        TAR_SIZE_GB=$(echo "scale=2; $TAR_SIZE_BYTES / 1024 / 1024 / 1024" | bc)
        echo "   ✅ Tar.gz: $YOUR_TAR"
        echo "      Size: $TAR_SIZE (${TAR_SIZE_GB} GB)"
        echo "      Owner: $(stat -c '%U' "$YOUR_TAR" 2>/dev/null || stat -f '%Su' "$YOUR_TAR" 2>/dev/null)"
    else
        echo "   ❌ Tar.gz not found"
    fi
    
    if [ -d "$YOUR_DIR" ]; then
        DIR_SIZE=$(du -sh "$YOUR_DIR" 2>/dev/null | cut -f1)
        DIR_SIZE_BYTES=$(du -sb "$YOUR_DIR" 2>/dev/null | cut -f1)
        DIR_SIZE_GB=$(echo "scale=2; $DIR_SIZE_BYTES / 1024 / 1024 / 1024" | bc)
        CELL_COUNT=$(find "$YOUR_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        echo "   ✅ Extracted directory: $YOUR_DIR"
        echo "      Size: $DIR_SIZE (${DIR_SIZE_GB} GB)"
        echo "      Cell directories: $CELL_COUNT"
    else
        echo "   ❌ Extracted directory not found (empty or not extracted)"
    fi
else
    echo "   ❌ Directory not found: $YOUR_BASE"
fi

echo ""
echo "3. Jens's dataset (ivf):"
echo ""

JENS_BASE="/staging/groups/bhaskar_group/ivf"
JENS_TAR="$JENS_BASE/embryo_dataset.tar.gz"
JENS_DIR="$JENS_BASE/embryo_dataset"

if [ -d "$JENS_BASE" ]; then
    echo "   Directory: $JENS_BASE"
    TOTAL_SIZE=$(du -sh "$JENS_BASE" 2>/dev/null | cut -f1)
    echo "   Total size: $TOTAL_SIZE"
    echo ""
    echo "   Contents:"
    ls -lh "$JENS_BASE" | head -10
    echo ""
    
    if [ -f "$JENS_TAR" ]; then
        TAR_SIZE=$(du -sh "$JENS_TAR" | cut -f1)
        TAR_SIZE_BYTES=$(du -sb "$JENS_TAR" | cut -f1)
        TAR_SIZE_GB=$(echo "scale=2; $TAR_SIZE_BYTES / 1024 / 1024 / 1024" | bc)
        echo "   ✅ Tar.gz: $JENS_TAR"
        echo "      Size: $TAR_SIZE (${TAR_SIZE_GB} GB)"
        echo "      Owner: $(stat -c '%U' "$JENS_TAR" 2>/dev/null || stat -f '%Su' "$JENS_TAR" 2>/dev/null)"
    else
        echo "   ❌ Tar.gz not found"
    fi
    
    if [ -d "$JENS_DIR" ]; then
        DIR_SIZE=$(du -sh "$JENS_DIR" 2>/dev/null | cut -f1)
        DIR_SIZE_BYTES=$(du -sb "$JENS_DIR" 2>/dev/null | cut -f1)
        DIR_SIZE_GB=$(echo "scale=2; $DIR_SIZE_BYTES / 1024 / 1024 / 1024" | bc)
        CELL_COUNT=$(find "$JENS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        echo "   ✅ Extracted directory: $JENS_DIR"
        echo "      Size: $DIR_SIZE (${DIR_SIZE_GB} GB)"
        echo "      Cell directories: $CELL_COUNT"
    else
        echo "   ❌ Extracted directory not found"
    fi
else
    echo "   ❌ Directory not found: $JENS_BASE"
    echo ""
    echo "   Searching for Jens's dataset in other locations..."
    
    # Search for jlundsgaard
    JENS_ALT="/staging/groups/bhaskar_group/jlundsgaard"
    if [ -d "$JENS_ALT" ]; then
        echo "   ✅ Found: $JENS_ALT"
        ls -lh "$JENS_ALT" | head -5
    fi
    
    # Search for any embryo_dataset in staging
    echo ""
    echo "   Searching for embryo_dataset in staging..."
    find /staging/groups/bhaskar_group/ -maxdepth 3 -name "embryo_dataset*" -o -name "*ivf*" 2>/dev/null | head -10 | while read item; do
        if [ -e "$item" ]; then
            if [ -d "$item" ]; then
                SIZE=$(du -sh "$item" 2>/dev/null | cut -f1)
                echo "     Directory: $item ($SIZE)"
            elif [ -f "$item" ]; then
                SIZE=$(du -sh "$item" 2>/dev/null | cut -f1)
                echo "     File: $item ($SIZE)"
            fi
        fi
    done
fi

echo ""
echo "4. Disk usage summary:"
echo ""

# Check disk space
df -h "$STAGING_BASE" 2>/dev/null || df -h /staging 2>/dev/null

echo ""
echo "5. Total sizes:"
echo ""

if [ -d "$YOUR_BASE" ]; then
    YOUR_TOTAL=$(du -sb "$YOUR_BASE" 2>/dev/null | cut -f1)
    YOUR_TOTAL_GB=$(echo "scale=2; $YOUR_TOTAL / 1024 / 1024 / 1024" | bc)
    echo "   Your dataset (rho9/ivf_data): ${YOUR_TOTAL_GB} GB"
fi

if [ -d "$JENS_BASE" ]; then
    JENS_TOTAL=$(du -sb "$JENS_BASE" 2>/dev/null | cut -f1)
    JENS_TOTAL_GB=$(echo "scale=2; $JENS_TOTAL / 1024 / 1024 / 1024" | bc)
    echo "   Jens's dataset (ivf): ${JENS_TOTAL_GB} GB"
fi

