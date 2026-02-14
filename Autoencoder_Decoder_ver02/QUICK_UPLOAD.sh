#!/bin/bash
# 快速上傳腳本 - 確保在正確的目錄中執行

cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

echo "當前目錄: $(pwd)"
echo ""
echo "檢查文件是否存在..."
if [ -f "extract_all_latent_trajectories.py" ] && [ -f "extract_latents.sh" ]; then
    echo "✓ 文件存在，開始上傳..."
    echo ""
    
    echo "1. 上傳 extract_all_latent_trajectories.py"
    scp extract_all_latent_trajectories.py rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/
    
    echo ""
    echo "2. 上傳 extract_latents.sh"
    scp extract_latents.sh rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/
    
    echo ""
    echo "============================================================"
    echo "✓ 上傳完成！"
    echo "============================================================"
else
    echo "✗ 錯誤：找不到文件"
    echo "請確認您在正確的目錄中"
    exit 1
fi








