#!/bin/bash
# Diagnose why job is held
# Usage: bash DIAGNOSE_HOLD.sh <cluster.proc>

if [ -z "$1" ]; then
    echo "=== Checking all held jobs ==="
    condor_q -hold
    echo ""
    echo "Usage: bash DIAGNOSE_HOLD.sh <cluster.proc>"
    echo "Example: bash DIAGNOSE_HOLD.sh 2609123.0"
    exit 1
fi

JOB_ID="$1"
echo "=== Diagnosing Hold Reason for Job: $JOB_ID ==="
echo ""

# 1. Check hold reason
echo "1. Hold Reason:"
condor_q -long $JOB_ID | grep -i "HoldReason\|HoldCode\|LastHoldReason" | head -5
echo ""

# 2. Check error log
echo "2. Error Log (last 50 lines):"
condor_tail -stderr $JOB_ID 2>/dev/null | tail -n 50
echo ""

# 3. Check output log
echo "3. Output Log (last 50 lines):"
condor_tail $JOB_ID 2>/dev/null | tail -n 50
echo ""

# 4. Check file transfer status
echo "4. File Transfer Status:"
condor_q -long $JOB_ID | grep -i "transfer\|file" | head -10
echo ""

# 5. Check resource requirements
echo "5. Resource Requirements:"
condor_q -long $JOB_ID | grep -i "request\|gpu\|memory\|disk" | head -10
echo ""

# 6. Check container/image
echo "6. Container/Image:"
condor_q -long $JOB_ID | grep -i "singularity\|container\|image" | head -5
echo ""

echo "=== Common Fixes ==="
echo ""
echo "If 'Transfer output files failure':"
echo "  - Check if run_train.sh creates results.tgz"
echo "  - Verify output directory exists"
echo ""
echo "If 'Failed to transfer input files':"
echo "  - Check all files in transfer_input_files exist"
echo "  - Verify file permissions"
echo ""
echo "If 'Container pull failed':"
echo "  - Try different PyTorch image tag"
echo ""
echo "To release hold:"
echo "  condor_release $JOB_ID"
echo ""
echo "To remove and resubmit:"
echo "  condor_rm $JOB_ID"
echo "  # Fix issue, then:"
echo "  condor_submit train_h200_lab.sub"





