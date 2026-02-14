#!/bin/bash
# 監控 HTCondor 作業的腳本

JOB_ID=${1:-2826350}

echo "============================================================"
echo "Monitoring HTCondor Job: $JOB_ID"
echo "============================================================"

# 查看作業狀態
echo ""
echo "1. Job Status:"
condor_q $JOB_ID

echo ""
echo "2. Job Details:"
condor_q -better-analyze $JOB_ID

echo ""
echo "3. Real-time Log (Ctrl+C to exit):"
echo "   Use: condor_tail -f $JOB_ID"
echo ""

# 檢查日誌文件
echo "4. Log Files:"
ls -lh ~/curvature_analysis_*.log 2>/dev/null || echo "   Log files not created yet"

echo ""
echo "5. To view logs:"
echo "   tail -f ~/curvature_analysis_ZS435-5.log"
echo "   tail -f ~/curvature_analysis_RS363-7.log"

