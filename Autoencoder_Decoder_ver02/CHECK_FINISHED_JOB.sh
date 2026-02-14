#!/bin/bash
# 檢查已完成任務的結果

echo "============================================================"
echo "任務已完成，檢查結果"
echo "============================================================"
echo ""
echo "請在 CHTC 上執行："
echo ""
echo "1. 查看任務歷史（看最終狀態）："
echo "   condor_history -limit 1 -submitter rho9 | grep extract"
echo ""
echo "2. 檢查結果目錄："
echo "   ls -lh /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/ 2>/dev/null || echo '目錄不存在'"
echo ""
echo "3. 檢查已提取的文件數量："
echo "   ls -1 /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/*.npy 2>/dev/null | wc -l"
echo ""
echo "4. 查看 metadata（看處理了多少個胚胎）："
echo "   cat /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/metadata.json 2>/dev/null || echo 'metadata 不存在'"
echo ""
echo "5. 查看最終的輸出日誌："
echo "   tail -100 /staging/groups/bhaskar_group/rho9/logs/extract_latents_v1_baseline.out 2>/dev/null || echo '日誌不存在'"
echo ""
echo "6. 查看錯誤日誌："
echo "   cat /staging/groups/bhaskar_group/rho9/logs/extract_latents_v1_baseline.err 2>/dev/null || echo '錯誤日誌不存在'"
echo ""
echo "============================================================"








