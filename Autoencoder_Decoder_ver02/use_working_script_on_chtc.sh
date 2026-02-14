#!/bin/bash
# 在 CHTC 上使用之前成功過的腳本（export_all_frame_latents_direct.py）

echo "============================================================"
echo "使用之前成功過的方法（export_all_frame_latents_direct.py）"
echo "============================================================"
echo ""
echo "這個方法之前成功提取了兩個細胞（一個好一個不好）"
echo "因為它直接在 Python 中處理 device，不需要 bash 變數"
echo ""
echo "步驟："
echo ""
echo "1. 上傳 export_all_frame_latents_direct.py 到 staging："
echo "   cd \"/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02\""
echo "   scp export_all_frame_latents_direct.py rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/"
echo ""
echo "2. 在 CHTC 上直接運行（如果數據路徑可訪問）："
echo "   python3 export_all_frame_latents_direct.py \\"
echo "     --checkpoint /staging/groups/bhaskar_group/rho9/checkpoints/checkpoint_epoch_50.pt \\"
echo "     --data_root /staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset \\"
echo "     --output /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents_all_frames_direct.npz"
echo ""
echo "或者創建一個簡單的 submit 文件來運行這個腳本"
echo ""
echo "============================================================"








