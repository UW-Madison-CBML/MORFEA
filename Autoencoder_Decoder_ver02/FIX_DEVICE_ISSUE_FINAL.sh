#!/bin/bash
# 最終修復 DEVICE 問題

echo "============================================================"
echo "任務再次因為 DEVICE 錯誤失敗"
echo "這表示 staging 上的 extract_latents.sh 還是舊版本"
echo "============================================================"
echo ""
echo "請執行以下步驟："
echo ""
echo "1. 在本地終端上傳修復後的 extract_latents.sh 到 staging："
echo "   cd \"/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02\""
echo "   scp extract_latents.sh rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/"
echo ""
echo "2. 在 CHTC 上確認文件已更新："
echo "   grep -A 5 'Ensure DEVICE' /staging/groups/bhaskar_group/rho9/extract_latents.sh"
echo "   # 應該看到 DEVICE 變量的檢查和默認值設置"
echo ""
echo "3. 確認修復已包含："
echo "   grep -A 3 'if \[ -z \"\\\$DEVICE\" \]' /staging/groups/bhaskar_group/rho9/extract_latents.sh"
echo ""
echo "4. 重新提交任務："
echo "   condor_submit ~/extract_latents_from_home.sub"
echo ""
echo "============================================================"
echo "這次應該會成功，因為："
echo "  ✅ extract_latents.sh 已修復 DEVICE 問題"
echo "  ✅ extract_latents_from_home.sub 已設置 GPUJobLength = long"
echo "  ✅ MaxJobRuntime = 86400 (24小時)"
echo "============================================================"








