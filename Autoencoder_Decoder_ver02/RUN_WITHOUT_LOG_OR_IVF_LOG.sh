#!/bin/bash
# 将日志保存到 ivf 目录，或者不使用日志文件

cd /staging/groups/bhaskar_group/rho9

# 检查当前进度
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
echo "已完成: $COUNT 个 plots"
echo ""

# 方案1: 日志保存到 ivf 目录
LOG_FILE="/staging/groups/bhaskar_group/ivf/tphate_batch.log"

echo "=== 方案1: 日志保存到 ivf 目录 ==="
nohup python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/ivf/aadhitya_v1_val \
    --knn 5 \
    --start_from $COUNT \
    --max_embryos 200 \
    > "$LOG_FILE" 2>&1 &

echo "✓ 作业已启动，PID: $!"
echo "日志文件: $LOG_FILE"
echo "查看日志: tail -f $LOG_FILE"
echo ""
echo "如果还是有问题，可以使用方案2（不使用日志文件）"






