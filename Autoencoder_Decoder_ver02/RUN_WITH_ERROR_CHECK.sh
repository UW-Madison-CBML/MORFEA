#!/bin/bash
# 运行并检查错误

cd /staging/groups/bhaskar_group/rho9

echo "=== 直接运行（查看错误信息）==="
echo ""

# 先直接运行一次，查看是否有错误
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/ivf/v1_baseline_tphate \
    --knn 5 \
    2>&1 | head -50

echo ""
echo "=== 如果有错误，请查看上面的输出 ==="
echo "如果没有错误，可以按 Ctrl+C 停止，然后后台运行"






