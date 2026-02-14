#!/bin/bash
# 将日志也保存到 staging 目录

cd /staging/groups/bhaskar_group/rho9

# 检查当前已完成的数量
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
echo "已完成: $COUNT 个 plots"
echo ""

# 日志文件也保存到 staging 目录
LOG_FILE="/staging/groups/bhaskar_group/rho9/tphate_batch_ivf.log"

echo "=== 保存到 ivf 目录，日志到 staging ==="
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






