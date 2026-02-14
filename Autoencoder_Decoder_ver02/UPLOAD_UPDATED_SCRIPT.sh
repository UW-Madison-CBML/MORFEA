#!/bin/bash
# 上传更新后的脚本（DPI 已降低到 150）

echo "=== 上传更新后的脚本 ==="
cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

scp generate_tphate_for_aadhitya.py rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/

echo ""
echo "✓ 上传完成！"
echo ""
echo "在 CHTC 上继续运行："
echo "cd /staging/groups/bhaskar_group/rho9"
echo "COUNT=\$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)"
echo "python3 generate_tphate_for_aadhitya.py \\"
echo "    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \\"
echo "    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \\"
echo "    --output_base /staging/groups/bhaskar_group/rho9/v1_baseline_tphate \\"
echo "    --knn 5 \\"
echo "    --start_from \$COUNT \\"
echo "    > /tmp/tphate_run.log 2>&1 &"






