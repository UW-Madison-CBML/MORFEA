#!/bin/bash
# Check completed/failed job result

JOBID=${1:-2651869}

echo "=== Checking Job $JOBID ==="
echo ""

# 1. Check job history
echo "=== Job History ==="
condor_history $JOBID -limit 1

echo ""
echo "=== Job Details ==="
condor_history $JOBID -limit 1 -long 2>/dev/null | grep -E "JobStatus|ExitCode|HoldReason|RemoteHost" | head -10

echo ""
echo "=== Log Files ==="
ls -lh logs/train_${JOBID}_0.* 2>/dev/null || echo "No log files found in logs/"

echo ""
echo "=== Output File (last 200 lines) ==="
if [ -f "logs/train_${JOBID}_0.out" ]; then
    tail -n 200 logs/train_${JOBID}_0.out
else
    echo "Output file not found: logs/train_${JOBID}_0.out"
    echo "Checking for any .out files:"
    ls -lh logs/*.out 2>/dev/null | tail -5
fi

echo ""
echo "=== Error File ==="
if [ -f "logs/train_${JOBID}_0.err" ]; then
    cat logs/train_${JOBID}_0.err
else
    echo "No error file found"
fi

echo ""
echo "=== Key Checkpoints ==="
if [ -f "logs/train_${JOBID}_0.out" ]; then
    echo "--- Dataset linking ---"
    grep -A 5 "Linking dataset\|Dataset symlink" logs/train_${JOBID}_0.out || echo "Not found"
    echo ""
    echo "--- Data directory check ---"
    grep -A 3 "Checking 'data' directory" logs/train_${JOBID}_0.out || echo "Not found"
    echo ""
    echo "--- Index building ---"
    grep -A 5 "FORCE (re)building\|After build_index" logs/train_${JOBID}_0.out || echo "Not found"
    echo ""
    echo "--- Training start ---"
    grep -A 3 "Starting training\|Loading dataset\|Epoch" logs/train_${JOBID}_0.out | head -10 || echo "Not found"
fi





