#!/bin/bash
# Quick status check script for CHTC training
# Usage: bash CHECK_STATUS.sh <cluster.proc>

if [ -z "$1" ]; then
    echo "Usage: bash CHECK_STATUS.sh <cluster.proc>"
    echo "Example: bash CHECK_STATUS.sh 2609123.0"
    echo ""
    echo "Or run without arguments to see all jobs:"
    condor_q
    exit 1
fi

JOB_ID="$1"
echo "=== Checking Job Status: $JOB_ID ==="
echo ""

# Check if job exists
condor_q $JOB_ID > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Job not found in queue. Checking history..."
    condor_history $JOB_ID
    echo ""
    echo "If job completed, check results:"
    echo "  ls -lh /staging/groups/bhaskar_group/ivf/results/"
    exit 0
fi

# Get job status
echo "1. Job Status:"
condor_q $JOB_ID
echo ""

# If running, show live output
STATUS=$(condor_q -format "%s\n" JobStatus $JOB_ID)
if [ "$STATUS" = "2" ]; then
    echo "2. Job is RUNNING. Live output (Ctrl+C to exit):"
    echo "---"
    condor_tail $JOB_ID
elif [ "$STATUS" = "1" ]; then
    echo "2. Job is IDLE (waiting for GPU). Checking why..."
    condor_q -better-analyze $JOB_ID
elif [ "$STATUS" = "5" ]; then
    echo "2. Job is HELD. Checking reason..."
    condor_q -hold
    condor_tail -stderr $JOB_ID
fi

echo ""
echo "3. Log files:"
ls -lh logs/train_${JOB_ID//./_}_*.{out,err,log} 2>/dev/null || echo "Log files not found yet"

echo ""
echo "4. Quick commands:"
echo "  View output:    condor_tail $JOB_ID"
echo "  View errors:     condor_tail -stderr $JOB_ID"
echo "  Cancel job:      condor_rm $JOB_ID"
echo "  Check results:   ls -lh /staging/groups/bhaskar_group/ivf/results/"





