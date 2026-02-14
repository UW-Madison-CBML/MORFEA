#!/bin/bash
# Debug job 2654961

JOB_ID="2654961.0"
JOB_NUM="2654961"

condor_history $JOB_NUM -limit 1

echo ""
condor_history $JOB_NUM -limit 1 -long | grep -E "JobStatus|ExitCode|HoldReason|RemoteHost|RemoteWallClockTime" | head -10

echo ""
cd ~/ivf/Raffael/2025-11-19 2>/dev/null || cd ~/ivf_train 2>/dev/null || pwd

echo ""
if [ -f "logs/train_${JOB_NUM}_0.err" ]; then
    ERR_SIZE=$(stat -f%z "logs/train_${JOB_NUM}_0.err" 2>/dev/null || stat -c%s "logs/train_${JOB_NUM}_0.err" 2>/dev/null || echo "0")
    if [ "$ERR_SIZE" != "0" ]; then
        cat "logs/train_${JOB_NUM}_0.err"
    else
    fi
else
fi

echo ""
if [ -f "logs/train_${JOB_NUM}_0.out" ]; then
    echo ""
    tail -n 100 "logs/train_${JOB_NUM}_0.out"
else
fi

echo ""
if [ -f "logs/train_${JOB_NUM}_0.log" ]; then
    echo ""
    tail -n 50 "logs/train_${JOB_NUM}_0.log"
else
fi

echo ""
if [ -f "logs/train_${JOB_NUM}_0.out" ]; then
fi

echo ""
if [ -f "logs/train_${JOB_NUM}_0.out" ]; then
fi

echo ""
if [ -d "checkpoints" ]; then
    ls -lh checkpoints/ | head -10
else
fi

if [ -f "logs/training_log.json" ]; then
    tail -n 20 logs/training_log.json
else
fi

echo ""
echo ""

