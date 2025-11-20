#!/bin/bash
set -e

echo "===== RUN_TRAIN START ====="
echo "CWD: $(pwd)"
echo "Initial files:"
ls

echo
echo "Step 1: 尋找並建立 data symlink"

# Try multiple possible paths
POSSIBLE_PATHS=(
    "/project/bhaskar_group/ivf"
    "/staging/groups/bhaskar_group/ivf"
    "/hdd/bhaskar_group/ivf"
    "/data/bhaskar_group/ivf"
)

DATA_PATH=""
for path in "${POSSIBLE_PATHS[@]}"; do
    echo "Checking: $path"
    if [ -d "$path" ]; then
        DATA_PATH="$path"
        echo "✓ Found data at: $path"
        # Check if it has content
        SAMPLE=$(ls "$path" 2>/dev/null | head -3)
        if [ -n "$SAMPLE" ]; then
            echo "  Sample contents: $SAMPLE"
        fi
        break
    else
        echo "  ✗ Not found"
    fi
done

if [ -z "$DATA_PATH" ]; then
    echo "✗ ERROR: Could not find data directory in any standard location"
    echo ""
    echo "Searching for alternative paths..."
    echo "Checking /staging:"
    ls -la /staging/ 2>&1 | head -10 || echo "  /staging/ does not exist"
    echo ""
    echo "Checking /hdd:"
    ls -la /hdd/ 2>&1 | head -10 || echo "  /hdd/ does not exist"
    echo ""
    echo "Searching for 'ivf' directories:"
    find /staging -maxdepth 3 -type d -name "*ivf*" 2>/dev/null | head -5 || echo "  No ivf in /staging"
    find /hdd -maxdepth 3 -type d -name "*ivf*" 2>/dev/null | head -5 || echo "  No ivf in /hdd"
    echo ""
    echo "Please contact lab admin for correct data path on GPU nodes"
    exit 1
fi

# Remove existing symlink if broken
[ -L data ] && rm data || true
ln -s "$DATA_PATH" data
echo "✓ Created symlink: data -> $DATA_PATH"

echo
echo "Check files after creating symlink:"
ls -l | head -10

echo
echo "Step 2: （選擇性）如果你希望先手動建 index.csv，可以在這裡跑 build_index.py"
# 只有在你有 build_index.py 的時候才打開下面這段：
# python -u build_index.py --root data --out index.csv
# ls -l index.csv

echo
echo "Step 3: 開始訓練"
python -u train.py 2>&1

echo "===== RUN_TRAIN DONE ====="
