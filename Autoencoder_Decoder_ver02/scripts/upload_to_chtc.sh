#!/bin/bash
# 上傳 curvature analysis 相關文件到 CHTC

CHTC_USER="rho9"
CHTC_HOST="ap2001.chtc.wisc.edu"
CHTC_DIR="/staging/groups/bhaskar_group/rho9/ivf_analysis"

echo "============================================================"
echo "Uploading files to CHTC..."
echo "============================================================"

# 上傳主腳本
echo "Uploading analyze_trajectory_curvature.py..."
scp scripts/analyze_trajectory_curvature.py \
    ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/scripts/

# 上傳運行腳本
echo "Uploading run_curvature_analysis.sh..."
scp scripts/run_curvature_analysis.sh \
    ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/scripts/

# 上傳單個分析腳本
echo "Uploading run_curvature_single.sh..."
scp scripts/run_curvature_single.sh \
    ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/scripts/

# 上傳 HTCondor submit 文件
echo "Uploading curvature_analysis_gpu.submit..."
scp scripts/curvature_analysis_gpu.submit \
    ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/

echo ""
echo "============================================================"
echo "✓ Upload complete!"
echo "============================================================"
echo ""
echo "Next steps on CHTC:"
echo "1. SSH to CHTC: ssh ${CHTC_USER}@${CHTC_HOST}"
echo "2. cd ${CHTC_DIR}"
echo "3. condor_submit curvature_analysis_gpu.submit"
echo ""

