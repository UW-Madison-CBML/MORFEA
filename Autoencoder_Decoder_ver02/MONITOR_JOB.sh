#!/bin/bash
# Monitor CHTC job and show output when it starts running

JOBID=${1:-$(condor_q rho9 -autoformat ClusterId 2>/dev/null | tail -1)}

if [ -z "$JOBID" ]; then
    echo "No job ID provided and no jobs found in queue"
    exit 1
fi

echo "=== Monitoring Job $JOBID ==="
echo ""

# Check job status
while true; do
    STATUS=$(condor_q $JOBID -autoformat JobStatus 2>/dev/null | tail -1)
    
    case "$STATUS" in
        0)
            echo "Job $JOBID: IDLE (waiting for resources...)"
            ;;
        1)
            echo "Job $JOBID: IDLE (waiting...)"
            ;;
        2)
            echo "Job $JOBID: RUNNING! Showing output..."
            echo ""
            condor_tail $JOBID.0
            break
            ;;
        3)
            echo "Job $JOBID: REMOVED"
            break
            ;;
        4)
            echo "Job $JOBID: COMPLETED"
            echo ""
            echo "=== Final Output ==="
            condor_tail $JOBID.0 2>/dev/null || echo "Output not available"
            break
            ;;
        5)
            echo "Job $JOBID: HELD"
            condor_q -long $JOBID.0 | grep -i "HoldReason" || echo "Check hold reason with: condor_q -long $JOBID.0 | grep HoldReason"
            break
            ;;
        *)
            echo "Job $JOBID: Status unknown ($STATUS)"
            break
            ;;
    esac
    
    sleep 5
done





