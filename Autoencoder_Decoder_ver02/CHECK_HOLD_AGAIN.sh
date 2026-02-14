#!/bin/bash
# Check why job is held again

JOB_ID="2650361.0"

echo "=== Checking Hold Reason ==="
echo ""

# 1. Check hold reason
echo "1. Hold Reason:"
condor_q -hold
echo ""

# 2. Detailed hold information
echo "2. Detailed Hold Information:"
condor_q -long $JOB_ID | grep -i "HoldReason\|HoldCode\|LastHoldReason" | head -5
echo ""

# 3. Check error log
echo "3. Error Log:"
if [ -f "logs/train_${JOB_ID//./_}_0.err" ]; then
    cat "logs/train_${JOB_ID//./_}_0.err"
else
    echo "Error log not found"
fi
echo ""

# 4. Check latest output
echo "4. Latest Output (last 30 lines):"
if [ -f "logs/train_${JOB_ID//./_}_0.out" ]; then
    tail -n 30 "logs/train_${JOB_ID//./_}_0.out"
else
    echo "Output log not found"
fi
echo ""

# 5. Check HTCondor log
echo "5. HTCondor Log (last 50 lines):"
if [ -f "logs/train_${JOB_ID//./_}_0.log" ]; then
    tail -n 50 "logs/train_${JOB_ID//./_}_0.log" | grep -i "hold\|error\|fail"
else
    echo "HTCondor log not found"
fi

echo ""
echo "=== Common Hold Reasons ==="
echo "1. Transfer output files failure - results.tgz not created"
echo "2. Script error - Python script failed"
echo "3. Missing files - dataset_ivf.py or index.csv not found"
echo "4. Out of memory - exceeded request_memory"
echo "5. Timeout - job ran too long"





