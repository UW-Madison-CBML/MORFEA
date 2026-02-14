#!/bin/bash
# 檢查任務為什麼立即完成

echo "============================================================"
echo "任務立即完成（可能是立即失敗）"
echo "============================================================"
echo ""
echo "請在 CHTC 上執行："
echo ""
echo "1. 查看任務歷史（看退出代碼）："
echo "   condor_history -limit 1 2853038 -long | grep -E 'ExitCode|ExitStatus|HoldReason'"
echo ""
echo "2. 查看錯誤日誌："
echo "   cat ~/logs/extract_latents_v1_baseline.err"
echo ""
echo "3. 查看輸出日誌："
echo "   cat ~/logs/extract_latents_v1_baseline.out"
echo ""
echo "4. 檢查 staging 上的日誌（如果有的話）："
echo "   tail -50 /staging/groups/bhaskar_group/rho9/logs/extract_latents_v1_baseline.out 2>/dev/null"
echo "   cat /staging/groups/bhaskar_group/rho9/logs/extract_latents_v1_baseline.err 2>/dev/null"
echo ""
echo "5. 檢查 condor 日誌："
echo "   grep '2853038' ~/logs/extract_latents_v1_baseline.log | tail -20"
echo ""
echo "============================================================"








