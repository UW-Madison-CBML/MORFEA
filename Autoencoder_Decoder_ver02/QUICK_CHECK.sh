#!/bin/bash
# Quick status check after fixing run_train.sh
# Usage: bash QUICK_CHECK.sh [job_id]

if [ -z "$1" ]; then
    echo "=== Current Job Status ==="
    condor_q
    echo ""
    echo "=== Held Jobs ==="
    condor_q -hold
    echo ""
    echo "Usage: bash QUICK_CHECK.sh <cluster.proc>"
    echo "Example: bash QUICK_CHECK.sh 2609485.0"
    exit 0
fi

JOB_ID="$1"
echo "=== Checking Job: $JOB_ID ==="
echo ""

# 1. Status
echo "1. Job Status:"
condor_q $JOB_ID 2>/dev/null || echo "Job not in queue (may have completed or been removed)"
echo ""

# 2. Hold reason (if held)
HELD=$(condor_q -hold -format "%s\n" ClusterId 2>/dev/null | grep -c "^${JOB_ID%%.*}$" || echo "0")
if [ "$HELD" != "0" ]; then
    echo "2. Hold Reason:"
    condor_q -long $JOB_ID | grep -i "HoldReason\|HoldCode" | head -3
    echo ""
fi

# 3. Recent output
echo "3. Recent Output (last 20 lines):"
if ls logs/train_${JOB_ID//./_}_*.out 1>/dev/null 2>&1; then
    tail -n 20 logs/train_${JOB_ID//./_}_*.out
else
    echo "Output log not found yet"
fi
echo ""

# 4. Recent errors
echo "4. Recent Errors (last 10 lines):"
if ls logs/train_${JOB_ID//./_}_*.err 1>/dev/null 2>&1; then
    tail -n 10 logs/train_${JOB_ID//./_}_*.err
else
    echo "Error log not found or empty"
fi
echo ""

# 5. Live output (if running)
STATUS=$(condor_q -format "%s\n" JobStatus $JOB_ID 2>/dev/null)
if [ "$STATUS" = "2" ]; then
    echo "5. Live Output (Ctrl+C to exit):"
    echo "---"
    condor_tail $JOB_ID
fi

echo ""
echo "=== Quick Commands ==="
echo "  View live:    condor_tail $JOB_ID"
echo "  View errors:   condor_tail -stderr $JOB_ID"
echo "  Cancel:        condor_rm $JOB_ID"
echo "  Check results: ls -lh /staging/groups/bhaskar_group/ivf/results/"


