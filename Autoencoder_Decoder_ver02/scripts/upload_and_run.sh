#!/bin/bash
# 上傳腳本並在 CHTC 上運行的完整流程

CHTC_USER="rho9"
CHTC_HOST="ap2001.chtc.wisc.edu"
CHTC_DIR="/staging/groups/bhaskar_group/rho9/ivf_analysis"

echo "============================================================"
echo "Step 1: Uploading optimized script to CHTC..."
echo "============================================================"

scp scripts/analyze_trajectory_curvature.py \
    ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/scripts/

if [ $? -eq 0 ]; then
    echo "✓ Upload successful!"
else
    echo "✗ Upload failed!"
    exit 1
fi

echo ""
echo "============================================================"
echo "Step 2: Next steps on CHTC:"
echo "============================================================"
echo ""
echo "1. SSH to CHTC:"
echo "   ssh ${CHTC_USER}@${CHTC_HOST}"
echo ""
echo "2. Run analysis:"
echo "   cd ${CHTC_DIR}"
echo "   bash scripts/run_curvature_analysis.sh"
echo ""
echo "Or run individually:"
echo "   python3 scripts/analyze_trajectory_curvature.py --video_name ZS435-5 --method pca --device cuda"
echo "   python3 scripts/analyze_trajectory_curvature.py --video_name RS363-7 --method pca --device cuda"
echo ""

