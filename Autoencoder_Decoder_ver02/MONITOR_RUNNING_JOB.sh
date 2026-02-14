#!/bin/bash
# Monitor running training job

JOB_ID="2650588.0"

echo "=== Job Status ==="
condor_q $JOB_ID

echo ""
echo "=== Real-time Output (last 50 lines) ==="
condor_tail $JOB_ID 2>/dev/null || echo "Output not available yet"

echo ""
echo "=== Error Log (if any) ==="
if [ -f "logs/train_2650588_0.err" ]; then
    cat logs/train_2650588_0.err
else
    echo "No error log yet"
fi

echo ""
echo "=== Output Log (last 30 lines) ==="
if [ -f "logs/train_2650588_0.out" ]; then
    tail -n 30 logs/train_2650588_0.out
else
    echo "Output log not created yet"
fi

echo ""
echo "=== To watch in real-time, run: ==="
echo "condor_tail -f $JOB_ID"
echo "or"
echo "tail -f logs/train_2650588_0.out"





