#!/bin/bash
# Watch job 2651891 and show output

JOBID=2651891

echo "=== Monitoring Job $JOBID ==="
echo ""

# Wait for job to start
echo "Waiting for job to start..."
while true; do
    STATUS=$(condor_q $JOBID -autoformat JobStatus 2>/dev/null | tail -1)
    
    if [ -z "$STATUS" ]; then
        echo ""
        echo "Job completed or not found. Checking output..."
        echo ""
        echo "=== Job History ==="
        condor_history $JOBID -limit 1
        echo ""
        echo "=== Output File (last 200 lines) ==="
        tail -n 200 logs/train_${JOBID}_0.out 2>/dev/null || echo "Output file not found"
        echo ""
        echo "=== Key Checkpoints ==="
        if [ -f "logs/train_${JOBID}_0.out" ]; then
            echo "--- run_train.sh start ---"
            grep -A 3 "Starting job\|CWD:" logs/train_${JOBID}_0.out | head -5
            echo ""
            echo "--- data symlink ---"
            grep -A 2 "data symlink" logs/train_${JOBID}_0.out
            echo ""
            echo "--- Building index.csv ---"
            grep -A 5 "Building index.csv\|Found.*cell directories\|Wrote index.csv" logs/train_${JOBID}_0.out
            echo ""
            echo "--- index.csv check ---"
            grep -A 2 "After build_index, check index.csv" logs/train_${JOBID}_0.out
            echo ""
            echo "--- Training start ---"
            grep -A 3 "Starting training\|Loading dataset\|Epoch" logs/train_${JOBID}_0.out | head -10
        fi
        break
    fi
    
    case "$STATUS" in
        0|1)
            echo -ne "\rStatus: IDLE (waiting...) - $(date +%H:%M:%S)"
            sleep 5
            ;;
        2)
            echo ""
            echo "✓ Job is RUNNING! Showing live output..."
            echo ""
            condor_tail $JOBID.0
            break
            ;;
        4)
            echo ""
            echo "Job COMPLETED. Showing final output..."
            echo ""
            tail -n 200 logs/train_${JOBID}_0.out 2>/dev/null || condor_tail $JOBID.0
            break
            ;;
        5)
            echo ""
            echo "⚠ Job HELD! Checking reason..."
            condor_q -long $JOBID.0 | grep -i "HoldReason" || echo "Run: condor_q -long $JOBID.0 | grep HoldReason"
            break
            ;;
        *)
            echo ""
            echo "Status: $STATUS"
            break
            ;;
    esac
done





