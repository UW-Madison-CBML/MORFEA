#!/bin/bash
# Debug why training is stuck after dependency installation

JOB_ID="2650361.0"

echo "=== Debugging Stuck Training ==="
echo ""

# 1. Check if job is still running
echo "1. Job Status:"
condor_q $JOB_ID
echo ""

# 2. Check error log
echo "2. Error Log:"
if [ -f "logs/train_${JOB_ID//./_}_0.err" ]; then
    ERR_SIZE=$(stat -f%z "logs/train_${JOB_ID//./_}_0.err" 2>/dev/null || stat -c%s "logs/train_${JOB_ID//./_}_0.err" 2>/dev/null || echo "0")
    if [ "$ERR_SIZE" != "0" ]; then
        echo "❌ ERRORS FOUND:"
        cat "logs/train_${JOB_ID//./_}_0.err"
    else
        echo "✅ No errors in error log"
    fi
else
    echo "Error log not found"
fi
echo ""

# 3. Check latest output
echo "3. Latest Output (last 50 lines):"
if [ -f "logs/train_${JOB_ID//./_}_0.out" ]; then
    tail -n 50 "logs/train_${JOB_ID//./_}_0.out"
else
    echo "Output log not found"
fi
echo ""

# 4. Check HTCondor log
echo "4. HTCondor Log (last 30 lines):"
if [ -f "logs/train_${JOB_ID//./_}_0.log" ]; then
    tail -n 30 "logs/train_${JOB_ID//./_}_0.log"
else
    echo "HTCondor log not found"
fi
echo ""

# 5. Check runtime
echo "5. Runtime Information:"
condor_q -long $JOB_ID | grep -E "RemoteWallClockTime|TotalJobRunTime|JobStartDate" | head -3
echo ""

# 6. Check what might be stuck
echo "6. Possible Issues:"
OUTPUT=$(tail -n 100 "logs/train_${JOB_ID//./_}_0.out" 2>/dev/null || echo "")

if echo "$OUTPUT" | grep -qi "error\|exception\|traceback\|failed"; then
    echo "❌ Found errors in output!"
    echo "$OUTPUT" | grep -i "error\|exception\|traceback\|failed" | tail -5
elif echo "$OUTPUT" | grep -qi "building index\|index.csv"; then
    echo "⏳ Still building index.csv (this can take time for large datasets)"
elif echo "$OUTPUT" | grep -qi "loading dataset"; then
    echo "⏳ Still loading dataset"
elif echo "$OUTPUT" | grep -qi "installing\|downloading"; then
    echo "⏳ Still installing dependencies"
else
    echo "⚠️  No clear progress indicator - may be stuck"
fi

echo ""
echo "=== Recommendations ==="
echo "If stuck for > 1 hour:"
echo "1. Check error log: cat logs/train_2650361_0.err"
echo "2. Check if dataset path is correct"
echo "3. Check if index.csv building is taking too long"
echo "4. Consider canceling and checking: condor_rm 2650361.0"





