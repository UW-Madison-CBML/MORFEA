#!/bin/bash
# Debug why training failed

JOB_ID="2650510.0"

echo "=== Debugging Training Failure ==="
echo ""

# 1. Check error log
echo "1. Error Log (CRITICAL):"
if [ -f "logs/train_${JOB_ID//./_}_0.err" ]; then
    cat "logs/train_${JOB_ID//./_}_0.err"
else
    echo "Error log not found"
fi
echo ""

# 2. Check output log
echo "2. Output Log (last 100 lines):"
if [ -f "logs/train_${JOB_ID//./_}_0.out" ]; then
    tail -n 100 "logs/train_${JOB_ID//./_}_0.out"
else
    echo "Output log not found"
fi
echo ""

# 3. Check what error occurred
echo "3. Searching for errors:"
if [ -f "logs/train_${JOB_ID//./_}_0.out" ]; then
    grep -i "error\|exception\|traceback\|failed\|not found" "logs/train_${JOB_ID//./_}_0.out" | tail -20
fi
echo ""

# 4. Check if dataset_ivf was found
echo "4. Checking dataset_ivf import:"
if [ -f "logs/train_${JOB_ID//./_}_0.out" ]; then
    grep -i "dataset_ivf\|import\|module" "logs/train_${JOB_ID//./_}_0.out" | tail -10
fi
echo ""

# 5. Check if index.csv was found
echo "5. Checking index.csv:"
if [ -f "logs/train_${JOB_ID//./_}_0.out" ]; then
    grep -i "index.csv\|file.*not found" "logs/train_${JOB_ID//./_}_0.out" | tail -10
fi

echo ""
echo "=== Common Issues ==="
echo "1. ModuleNotFoundError: dataset_ivf - PYTHONPATH issue"
echo "2. FileNotFoundError: index.csv - need to build it"
echo "3. FileNotFoundError: data directory - wrong path"
echo "4. CUDA out of memory - need more memory or smaller batch"





