#!/bin/bash
# Final fix for dataset_ivf.py and submit file

cd ~/ivf_train

echo "=== Final Fix ==="
echo ""

# 1. Fix dataset_ivf.py (copy from repo, not download)
echo "1. Fixing dataset_ivf.py..."
if [ -f ~/ivf/dataset_ivf.py ]; then
    cp ~/ivf/dataset_ivf.py .
    echo "✅ Copied from ~/ivf/dataset_ivf.py"
elif [ -f ~/ivf/Code/dataset_ivf.py ]; then
    cp ~/ivf/Code/dataset_ivf.py .
    echo "✅ Copied from ~/ivf/Code/dataset_ivf.py"
else
    echo "❌ dataset_ivf.py not found in repo!"
    echo "   Searching..."
    find ~/ivf -name "dataset_ivf.py" 2>/dev/null
fi

# Verify size
if [ -f "dataset_ivf.py" ]; then
    SIZE=$(stat -f%z "dataset_ivf.py" 2>/dev/null || stat -c%s "dataset_ivf.py" 2>/dev/null || echo "0")
    if [ "$SIZE" -lt 100 ]; then
        echo "❌ dataset_ivf.py too small ($SIZE bytes)!"
        echo "   Need to copy from repo or download correctly"
    else
        echo "✅ dataset_ivf.py: $SIZE bytes"
    fi
fi
echo ""

# 2. Fix duplicate in submit file
echo "2. Fixing duplicate in submit file..."
sed -i 's/dataset_ivf.py, dataset_ivf.py/dataset_ivf.py/' train_h200_lab.sub
sed -i 's/, dataset_ivf.py, dataset_ivf.py/, dataset_ivf.py/' train_h200_lab.sub
echo "✅ Fixed duplicates"
echo ""

# 3. Verify submit file
echo "3. Submit file:"
grep "transfer_input_files" train_h200_lab.sub
echo ""

# 4. Remove old held job
echo "4. Removing old held job..."
condor_rm 2650507.0 2>/dev/null || echo "Job already removed or not found"
echo ""

# 5. Verify all files
echo "5. Files ready:"
ls -lh dataset_ivf.py build_index.py run_train.sh train_h200_lab.sub 2>/dev/null
echo ""

echo "=== Ready ==="
echo "New job 2650510 is already submitted and idle"
echo "Check status: condor_q"





