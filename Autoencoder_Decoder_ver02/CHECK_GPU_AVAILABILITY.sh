#!/bin/bash
# Check GPU availability and queue status

echo "=== GPU Availability Check ==="
echo ""

# 1. Check available H200 GPUs
echo "1. Available H200 GPUs:"
condor_status -constraint 'regexp("H200", CUDADeviceName)' -af Name State Activity CUDADeviceName TotalGPUs 2>/dev/null | head -10
echo ""

# 2. Check lab GPU nodes
echo "2. Bhaskar Lab GPU Nodes:"
condor_status -constraint 'regexp("bhaskar", Machine)' -af Name State Activity TotalGPUs GPUs 2>/dev/null | head -10
echo ""

# 3. Check your job status
echo "3. Your Job Status:"
condor_q -format "Job ID: %s.%s\n" ClusterId ProcId -format "Status: %s\n" JobStatus -format "Priority: %d\n" JobPrio
echo ""

# 4. Check queue position (if job is idle)
JOB_ID=$(condor_q -format "%s.%s\n" ClusterId ProcId | head -1)
if [ -n "$JOB_ID" ]; then
    echo "4. Detailed Job Analysis:"
    condor_q -better-analyze $JOB_ID 2>/dev/null | head -30
fi

echo ""
echo "=== Tips ==="
echo "- If all GPUs are 'Busy', they're in use"
echo "- If GPUs are 'Idle', your job should start soon"
echo "- Lab jobs typically get priority over general pool"
echo "- Wait time: 0-4 hours typically"





