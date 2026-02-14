#!/bin/bash
# Check if data paths are pointing to the correct location (rho9's data, not Jens's)

echo "=== Checking Data Paths ==="
echo ""

# Check current data symlink/directory
echo "1. Current 'data' location:"
if [ -L "data" ]; then
    REAL_PATH=$(readlink -f data 2>/dev/null || readlink data)
    echo "   'data' is a symlink pointing to: $REAL_PATH"
elif [ -d "data" ]; then
    REAL_PATH=$(cd data && pwd)
    echo "   'data' is a directory at: $REAL_PATH"
else
    echo "   ❌ 'data' not found"
fi

echo ""

# Check if it's pointing to Jens's old data
if [[ "$REAL_PATH" == *"/ivf/"* ]] && [[ "$REAL_PATH" != *"/rho9/"* ]]; then
    echo "⚠️  WARNING: 'data' is pointing to Jens's old dataset!"
    echo "   Path: $REAL_PATH"
    echo "   This is the OLD dataset (jlundsgaard's project)"
    echo ""
    echo "   You should use YOUR dataset at:"
    echo "   /staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset"
fi

echo ""

# Check both staging locations
echo "2. Checking staging locations:"
echo ""

# Jens's old location
JENS_PATH="/staging/groups/bhaskar_group/ivf/embryo_dataset"
if [ -d "$JENS_PATH" ]; then
    JENS_SIZE=$(du -sh "$JENS_PATH" 2>/dev/null | cut -f1)
    JENS_CELLS=$(find "$JENS_PATH" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "   Jens's old dataset:"
    echo "     Path: $JENS_PATH"
    echo "     Size: $JENS_SIZE"
    echo "     Cell directories: $JENS_CELLS"
else
    echo "   Jens's old dataset: Not found at $JENS_PATH"
fi

echo ""

# Your new location
YOUR_PATH="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset"
if [ -d "$YOUR_PATH" ]; then
    YOUR_SIZE=$(du -sh "$YOUR_PATH" 2>/dev/null | cut -f1)
    YOUR_CELLS=$(find "$YOUR_PATH" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "   Your dataset:"
    echo "     Path: $YOUR_PATH"
    echo "     Size: $YOUR_SIZE"
    echo "     Cell directories: $YOUR_CELLS"
else
    echo "   Your dataset: Not found at $YOUR_PATH"
    echo "   (May need to extract from tar.gz)"
fi

echo ""

# Check tar.gz files
echo "3. Checking tar.gz files:"
echo ""

JENS_TAR="/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz"
YOUR_TAR="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"

if [ -f "$JENS_TAR" ]; then
    JENS_TAR_SIZE=$(du -sh "$JENS_TAR" 2>/dev/null | cut -f1)
    echo "   Jens's tar.gz: $JENS_TAR_SIZE at $JENS_TAR"
else
    echo "   Jens's tar.gz: Not found"
fi

if [ -f "$YOUR_TAR" ]; then
    YOUR_TAR_SIZE=$(du -sh "$YOUR_TAR" 2>/dev/null | cut -f1)
    echo "   Your tar.gz: $YOUR_TAR_SIZE at $YOUR_TAR"
else
    echo "   Your tar.gz: Not found"
fi

echo ""
echo "=== Summary ==="
echo ""

if [[ "$REAL_PATH" == *"/ivf/"* ]] && [[ "$REAL_PATH" != *"/rho9/"* ]]; then
    echo "❌ PROBLEM: You're using Jens's old dataset!"
    echo ""
    echo "🔧 SOLUTION:"
    echo "   1. Remove old symlink:"
    echo "      rm -f data"
    echo ""
    echo "   2. Extract YOUR dataset from YOUR tar.gz:"
    echo "      mkdir -p ~/ivf_repo/data_raw"
    echo "      cd ~/ivf_repo/data_raw"
    echo "      tar -xzvf $YOUR_TAR"
    echo ""
    echo "   3. Create symlink to YOUR dataset:"
    echo "      cd ~/ivf_repo"
    echo "      ln -s data_raw/embryo_dataset data"
    echo ""
    echo "   4. Verify:"
    echo "      ls -lh data"
    echo "      du -sh data"
else
    echo "✅ Path looks correct (using rho9's dataset)"
    echo ""
    if [ -d "$YOUR_PATH" ] && [ "$YOUR_CELLS" -lt 10 ]; then
        echo "⚠️  But extracted dataset seems incomplete ($YOUR_CELLS cells)"
        echo "   You may need to re-extract from tar.gz"
    fi
fi

