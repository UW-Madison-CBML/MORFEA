#!/bin/bash
# Monitor job 2654961

JOB_ID="2654961.0"

echo "=== Job Status ==="
condor_q $JOB_ID

echo ""
echo "=== Real-time Output (last 50 lines) ==="
condor_tail $JOB_ID 2>/dev/null || echo "Output not available yet (job may still be starting)"

echo ""
echo "=== Error Log (if any) ==="
if [ -f "logs/train_2654961_0.err" ]; then
    ERR_SIZE=$(stat -f%z "logs/train_2654961_0.err" 2>/dev/null || stat -c%s "logs/train_2654961_0.err" 2>/dev/null || echo "0")
    if [ "$ERR_SIZE" != "0" ]; then
        tail -n 30 "logs/train_2654961_0.err"
    else
        echo "No errors (good!)"
    fi
else
    echo "Error log not created yet"
fi

echo ""
echo "=== Output Log (last 30 lines) ==="
if [ -f "logs/train_2654961_0.out" ]; then
    tail -n 30 "logs/train_2654961_0.out"
else
    echo "Output log not created yet"
fi

echo ""
echo "=== Quick Commands ==="
echo "Watch real-time: condor_tail -f $JOB_ID"
echo "Check status: condor_q $JOB_ID"
echo "View errors: condor_tail -stderr $JOB_ID"
echo "Cancel job: condor_rm $JOB_ID"

