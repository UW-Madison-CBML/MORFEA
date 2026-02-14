#!/bin/bash
# Watch a specific CHTC job and show output when it starts

JOBID=${1:-2651869}

echo "=== Monitoring Job $JOBID ==="
echo ""

# Function to check job status
check_status() {
    STATUS=$(condor_q $JOBID -autoformat JobStatus 2>/dev/null | tail -1)
    case "$STATUS" in
        0|1) echo "IDLE (waiting for resources...)" ;;
        2) echo "RUNNING" ;;
        3) echo "REMOVED" ;;
        4) echo "COMPLETED" ;;
        5) echo "HELD" ;;
        *) echo "UNKNOWN ($STATUS)" ;;
    esac
    return $STATUS
}

# Wait for job to start
echo "Waiting for job to start..."
while true; do
    STATUS=$(condor_q $JOBID -autoformat JobStatus 2>/dev/null | tail -1)
    
    if [ -z "$STATUS" ]; then
        echo "Job not found in queue. Checking history..."
        condor_history $JOBID -limit 1
        echo ""
        echo "=== Final Output ==="
        tail -n 200 logs/train_${JOBID}_0.out 2>/dev/null || echo "Output file not found"
        echo ""
        echo "=== Errors ==="
        cat logs/train_${JOBID}_0.err 2>/dev/null || echo "No errors"
        break
    fi
    
    case "$STATUS" in
        0|1)
            echo -ne "\rStatus: IDLE (waiting...) - $(date +%H:%M:%S)"
            sleep 5
            ;;
        2)
            echo ""
            echo "✓ Job is RUNNING! Showing output..."
            echo ""
            condor_tail $JOBID.0
            break
            ;;
        4)
            echo ""
            echo "Job COMPLETED. Showing final output..."
            echo ""
            tail -n 200 logs/train_${JOBID}_0.out 2>/dev/null || condor_tail $JOBID.0
            echo ""
            echo "=== Errors ==="
            cat logs/train_${JOBID}_0.err 2>/dev/null || echo "No errors"
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





