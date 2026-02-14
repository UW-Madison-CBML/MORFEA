#!/bin/bash
# Upload curvature analysis script to CHTC group directory

echo "=== Uploading Curvature Analysis Script to CHTC Group Directory ==="
echo ""

CHTC_USER="rho9"
CHTC_HOST="ap2001.chtc.wisc.edu"
GROUP_SCRIPTS_DIR="/staging/groups/bhaskar_group/rho9/ivf_analysis/scripts"
LOCAL_SCRIPT="scripts/analyze_trajectory_curvature.py"

# Check if script exists locally
if [ ! -f "$LOCAL_SCRIPT" ]; then
    echo "❌ Error: Script not found at $LOCAL_SCRIPT"
    echo "   Make sure you're running from the project root directory"
    exit 1
fi

echo "Local script: $LOCAL_SCRIPT"
echo "Remote destination: $CHTC_USER@$CHTC_HOST:$GROUP_SCRIPTS_DIR/"
echo ""

# Create remote directory first
echo "Creating remote directory..."
ssh $CHTC_USER@$CHTC_HOST "mkdir -p $GROUP_SCRIPTS_DIR"

# Upload script
echo "Uploading script..."
scp "$LOCAL_SCRIPT" $CHTC_USER@$CHTC_HOST:$GROUP_SCRIPTS_DIR/

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Upload successful!"
    echo ""
    echo "Next steps:"
    echo "  1. SSH to CHTC:"
    echo "     ssh $CHTC_USER@$CHTC_HOST"
    echo ""
    echo "  2. Navigate to group directory:"
    echo "     cd /staging/groups/bhaskar_group/rho9/ivf_analysis"
    echo ""
    echo "  3. Run analysis:"
    echo "     python3 scripts/analyze_trajectory_curvature.py --video_name RS363-7"
else
    echo ""
    echo "❌ Upload failed"
    exit 1
fi

