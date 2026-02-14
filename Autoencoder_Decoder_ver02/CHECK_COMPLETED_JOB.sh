#!/bin/bash
# Check completed/failed job status and output

JOBID=${1:-2651859}

echo "=== Checking Job $JOBID ==="
echo ""

# 1. Check job history
echo "=== Job History ==="
condor_history $JOBID -limit 1

echo ""
echo "=== Job Details ==="
condor_history $JOBID -limit 1 -long | grep -E "JobStatus|ExitCode|HoldReason|RemoteHost" | head -10

echo ""
echo "=== Log Files ==="
ls -lh logs/train_${JOBID}_0.* 2>/dev/null || echo "No log files found in logs/"

echo ""
echo "=== Output File ==="
if [ -f "logs/train_${JOBID}_0.out" ]; then
    echo "Last 100 lines of output:"
    tail -n 100 logs/train_${JOBID}_0.out
else
    echo "Output file not found: logs/train_${JOBID}_0.out"
fi

echo ""
echo "=== Error File ==="
if [ -f "logs/train_${JOBID}_0.err" ]; then
    echo "Error output:"
    cat logs/train_${JOBID}_0.err
else
    echo "No error file found"
fi

echo ""
echo "=== Log File (condor log) ==="
if [ -f "logs/train_${JOBID}_0.log" ]; then
    echo "Last 50 lines:"
    tail -n 50 logs/train_${JOBID}_0.log
fi





