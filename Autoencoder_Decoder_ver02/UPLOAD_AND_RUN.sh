#!/bin/bash
# 上传脚本并运行第一批

echo "=== 上传更新后的脚本 ==="
cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

if [ -f generate_tphate_for_aadhitya.py ]; then
    echo "上传脚本..."
    scp generate_tphate_for_aadhitya.py rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/
    echo "✓ 上传完成"
else
    echo "⚠️  脚本文件不存在"
    exit 1
fi

echo ""
echo "=== 在 CHTC 上运行以下命令 ==="
echo ""
echo "cd /staging/groups/bhaskar_group/rho9"
echo ""
echo "# 第一批：处理胚胎 183-382 (200个)"
echo "nohup python3 generate_tphate_for_aadhitya.py \\"
echo "    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \\"
echo "    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \\"
echo "    --output_base aadhitya_v1_val \\"
echo "    --knn 5 \\"
echo "    --start_from 182 \\"
echo "    --max_embryos 200 \\"
echo "    > tphate_batch1.log 2>&1 &"
echo ""
echo "tail -f tphate_batch1.log"






