#!/bin/bash
# 上傳修復後的 extract_latents.sh

echo "============================================================"
echo "上傳修復後的 extract_latents.sh 到 staging"
echo "============================================================"
echo ""
echo "問題：DEVICE 變量仍然是空的"
echo "解決：需要重新上傳包含 DEVICE 修復的腳本"
echo ""
echo "請在本地終端執行："
echo ""
echo "cd \"/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02\""
echo "scp extract_latents.sh rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/"
echo ""
echo "然後在 CHTC 上確認："
echo "grep -A 5 'Ensure DEVICE' /staging/groups/bhaskar_group/rho9/extract_latents.sh"
echo ""
echo "應該看到："
echo "  # Ensure DEVICE has a valid value..."
echo "  if [ -z \"\$DEVICE\" ]; then"
echo "    echo \"Warning: Could not detect device, defaulting to cuda...\""
echo "    DEVICE=\"cuda\""
echo "  fi"
echo ""
echo "============================================================"








