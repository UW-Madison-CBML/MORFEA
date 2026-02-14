#!/bin/bash
# 不使用日志文件（直接运行，不保存日志）

cd /staging/groups/bhaskar_group/rho9

# 检查当前进度
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
echo "已完成: $COUNT 个 plots"
echo ""

echo "=== 不使用日志文件，直接运行 ==="
echo "输出保存到: /staging/groups/bhaskar_group/ivf/aadhitya_v1_val"
echo ""

# 不使用 nohup，直接在后台运行，输出到 /dev/null
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/ivf/aadhitya_v1_val \
    --knn 5 \
    --start_from $COUNT \
    --max_embryos 200 \
    > /dev/null 2>&1 &

echo "✓ 作业已在后台运行，PID: $!"
echo "检查进度: ls -1 /staging/groups/bhaskar_group/ivf/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l"






