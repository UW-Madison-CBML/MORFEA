#!/bin/bash
# Check what happened to a job that's no longer in queue
# Usage: bash CHECK_WHAT_HAPPENED.sh [job_id]

echo "=== Diagnosing Completed/Removed Job ==="
echo ""

# If job ID provided, focus on that
if [ -n "$1" ]; then
    JOB_ID="$1"
    CLUSTER="${JOB_ID%%.*}"
    PROC="${JOB_ID##*.}"
    echo "Checking job: $JOB_ID"
    echo ""
else
    echo "No job ID provided, checking recent history..."
    echo ""
    CLUSTER=""
    PROC=""
fi

# 1. Check job history
echo "1. Recent Job History:"
if [ -n "$CLUSTER" ]; then
    condor_history -constraint "ClusterId == $CLUSTER" -limit 1
else
    condor_history -limit 5
fi
echo ""

# 2. Check results in staging
echo "2. Results in Staging:"
if [ -n "$CLUSTER" ]; then
    ls -lh /staging/groups/bhaskar_group/ivf/results/results_${CLUSTER}_${PROC}.tgz 2>/dev/null && \
        echo "✅ Results file found!" || \
        echo "❌ Results file not found"
else
    ls -lht /staging/groups/bhaskar_group/ivf/results/ 2>/dev/null | head -5 || \
        echo "No results directory or empty"
fi
echo ""

# 3. Check local logs
echo "3. Local Log Files:"
if [ -n "$CLUSTER" ]; then
    LOG_PATTERN="train_${CLUSTER}_${PROC}"
    ls -lh logs/${LOG_PATTERN}.* 2>/dev/null || echo "No log files found for this job"
else
    ls -lht logs/ | head -5
fi
echo ""

# 4. Check last output
echo "4. Last Output (if available):"
if [ -n "$CLUSTER" ]; then
    if [ -f "logs/train_${CLUSTER}_${PROC}.out" ]; then
        echo "--- Last 30 lines of output ---"
        tail -n 30 logs/train_${CLUSTER}_${PROC}.out
    else
        echo "Output log not found"
    fi
else
    LATEST_OUT=$(ls -t logs/train_*.out 2>/dev/null | head -1)
    if [ -n "$LATEST_OUT" ]; then
        echo "--- Last 30 lines of latest output ---"
        tail -n 30 "$LATEST_OUT"
    else
        echo "No output logs found"
    fi
fi
echo ""

# 5. Check for errors
echo "5. Errors (if any):"
if [ -n "$CLUSTER" ]; then
    if [ -f "logs/train_${CLUSTER}_${PROC}.err" ]; then
        ERR_SIZE=$(stat -f%z "logs/train_${CLUSTER}_${PROC}.err" 2>/dev/null || stat -c%s "logs/train_${CLUSTER}_${PROC}.err" 2>/dev/null || echo "0")
        if [ "$ERR_SIZE" != "0" ]; then
            echo "--- Error log content ---"
            cat logs/train_${CLUSTER}_${PROC}.err
        else
            echo "Error log is empty (good!)"
        fi
    else
        echo "No error log found"
    fi
else
    LATEST_ERR=$(ls -t logs/train_*.err 2>/dev/null | head -1)
    if [ -n "$LATEST_ERR" ]; then
        ERR_SIZE=$(stat -f%z "$LATEST_ERR" 2>/dev/null || stat -c%s "$LATEST_ERR" 2>/dev/null || echo "0")
        if [ "$ERR_SIZE" != "0" ]; then
            echo "--- Latest error log ---"
            cat "$LATEST_ERR"
        else
            echo "Latest error log is empty (good!)"
        fi
    else
        echo "No error logs found"
    fi
fi
echo ""

# 6. Check if training ran
echo "6. Training Evidence:"
if [ -n "$CLUSTER" ]; then
    if grep -qi "epoch\|loss\|training" logs/train_${CLUSTER}_${PROC}.out 2>/dev/null; then
        echo "✅ Training output found in logs"
        grep -i "epoch\|loss" logs/train_${CLUSTER}_${PROC}.out | tail -5
    else
        echo "❌ No training output found"
    fi
else
    if grep -qi "epoch\|loss\|training" logs/train_*.out 2>/dev/null; then
        echo "✅ Training output found"
        grep -i "epoch\|loss" logs/train_*.out | tail -5
    else
        echo "❌ No training output found"
    fi
fi
echo ""

# 7. Summary
echo "=== Summary ==="
if [ -n "$CLUSTER" ]; then
    if [ -f "/staging/groups/bhaskar_group/ivf/results/results_${CLUSTER}_${PROC}.tgz" ]; then
        echo "✅ Job likely completed successfully!"
        echo "   Results: /staging/groups/bhaskar_group/ivf/results/results_${CLUSTER}_${PROC}.tgz"
    elif grep -qi "epoch" logs/train_${CLUSTER}_${PROC}.out 2>/dev/null; then
        echo "⚠️  Training started but may have failed"
        echo "   Check error log for details"
    else
        echo "❌ Job failed to start or failed immediately"
        echo "   Check error log for details"
    fi
else
    echo "Run with job ID for specific diagnosis:"
    echo "  bash CHECK_WHAT_HAPPENED.sh <cluster.proc>"
fi





