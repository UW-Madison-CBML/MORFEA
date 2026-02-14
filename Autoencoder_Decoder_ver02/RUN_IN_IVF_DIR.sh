#!/bin/bash
# 保存到 ivf 目录（可能没有配额限制）

cd /staging/groups/bhaskar_group/rho9

echo "=== 检查当前进度 ==="
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
echo "已完成: $COUNT 个 plots"

echo ""
echo "=== 保存到 ivf 目录（可能没有配额限制）==="
echo "输出目录: /staging/groups/bhaskar_group/ivf/aadhitya_v1_val"
echo ""

# 从当前进度继续（假设已完成 COUNT 个）
nohup python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/ivf/aadhitya_v1_val \
    --knn 5 \
    --start_from $COUNT \
    --max_embryos 200 \
    > tphate_batch_ivf.log 2>&1 &

echo "✓ 作业已启动，PID: $!"
echo "查看日志: tail -f tphate_batch_ivf.log"






