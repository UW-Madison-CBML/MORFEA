#!/bin/bash
# 上傳修改後的文件到 CHTC

echo "============================================================"
echo "上傳修改後的文件到 CHTC"
echo "============================================================"
echo ""
echo "已套用成功方法的邏輯："
echo "  ✓ extract_all_latent_trajectories.py - Python 自動檢測 device"
echo "  ✓ extract_latents.sh - 移除了 --device 參數傳遞"
echo ""

cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

echo "1. 上傳 extract_all_latent_trajectories.py"
scp extract_all_latent_trajectories.py rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/

echo ""
echo "2. 上傳 extract_latents.sh"
scp extract_latents.sh rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/

echo ""
echo "============================================================"
echo "✓ 文件上傳完成"
echo ""
echo "下一步："
echo "  1. SSH 到 CHTC: ssh rho9@ap2001.chtc.wisc.edu"
echo "  2. 確認文件已更新: ls -lh /staging/groups/bhaskar_group/rho9/extract_*.py /staging/groups/bhaskar_group/rho9/extract_*.sh"
echo "  3. 重新提交任務: condor_submit ~/extract_latents_from_home.sub"
echo "============================================================"








