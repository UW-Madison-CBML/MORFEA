#!/bin/bash
# 檢查是否有部分結果

echo "============================================================"
echo "檢查是否有部分結果（任務運行12小時後被終止）"
echo "============================================================"
echo ""
echo "請在 CHTC 上執行："
echo ""
echo "1. 檢查結果目錄："
echo "   ls -lh /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/ 2>/dev/null || echo '目錄不存在'"
echo ""
echo "2. 檢查已提取的文件數量："
echo "   ls -1 /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/*.npy 2>/dev/null | wc -l"
echo ""
echo "3. 查看已提取的文件列表："
echo "   ls -lh /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/*.npy 2>/dev/null | head -10"
echo ""
echo "4. 檢查 metadata（看處理了多少個胚胎）："
echo "   cat /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/metadata.json 2>/dev/null || echo 'metadata 不存在'"
echo ""
echo "============================================================"

