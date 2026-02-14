#!/bin/bash
# 分批运行 T-PHATE 生成

cd /staging/groups/bhaskar_group/rho9

# 停止之前的进程
pkill -f "generate_tphate_for_aadhitya.py" 2>/dev/null

echo "=== 分批运行 T-PHATE 生成 ==="
echo "已完成的胚胎: 182"
echo "总胚胎数: 704"
echo "剩余: 522 个"
echo ""

# 第一批：处理接下来的 200 个（从第 183 个开始，索引是 182）
echo "第一批：处理胚胎 183-382 (200个)..."
nohup python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_val \
    --knn 5 \
    --start_from 182 \
    --max_embryos 200 \
    > tphate_batch1.log 2>&1 &

echo "✓ 第一批已启动，PID: $!"
echo "查看日志: tail -f tphate_batch1.log"
echo ""
echo "等待第一批完成后，运行第二批："
echo "  nohup python3 generate_tphate_for_aadhitya.py \\"
echo "    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \\"
echo "    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \\"
echo "    --output_base aadhitya_v1_val \\"
echo "    --knn 5 \\"
echo "    --start_from 382 \\"
echo "    --max_embryos 200 \\"
echo "    > tphate_batch2.log 2>&1 &"






