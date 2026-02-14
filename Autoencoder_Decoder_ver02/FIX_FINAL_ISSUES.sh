#!/bin/bash
# Fix final issues: build_index.py and duplicate dataset_ivf.py

cd ~/ivf_train

echo "=== Fixing Final Issues ==="
echo ""

# 1. Fix build_index.py (copy from repo)
echo "1. Fixing build_index.py..."
if [ -f ~/ivf/build_index.py ]; then
    cp ~/ivf/build_index.py .
    echo "✅ Copied from ~/ivf/build_index.py"
elif [ -f ~/ivf/Code/build_index.py ]; then
    cp ~/ivf/Code/build_index.py .
    echo "✅ Copied from ~/ivf/Code/build_index.py"
else
    echo "⚠️  build_index.py not found in repo"
    echo "   Will try to build index during training or use existing"
fi

# Verify size
if [ -f "build_index.py" ]; then
    SIZE=$(stat -f%z "build_index.py" 2>/dev/null || stat -c%s "build_index.py" 2>/dev/null || echo "0")
    if [ "$SIZE" -lt 100 ]; then
        echo "❌ build_index.py too small ($SIZE bytes), removing..."
        rm -f build_index.py
    else
        echo "✅ build_index.py: $SIZE bytes"
    fi
fi
echo ""

# 2. Fix duplicate dataset_ivf.py in submit file
echo "2. Fixing duplicate in submit file..."
sed -i 's/dataset_ivf.py, dataset_ivf.py/dataset_ivf.py/' train_h200_lab.sub
echo "✅ Removed duplicate"
echo ""

# 3. Verify submit file
echo "3. Submit file transfer_input_files:"
grep "transfer_input_files" train_h200_lab.sub
echo ""

# 4. Verify all files
echo "4. Files ready:"
ls -lh dataset_ivf.py build_index.py run_train.sh train_h200_lab.sub 2>/dev/null | grep -v "cannot access"
echo ""

echo "=== Ready to Resubmit ==="
echo "Next: condor_submit train_h200_lab.sub"





