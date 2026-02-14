#!/bin/bash
# 简单运行 T-PHATE 生成（不使用 --skip_existing）

cd /staging/groups/bhaskar_group/rho9

# 停止之前的进程（如果有）
pkill -f "generate_tphate_for_aadhitya.py" 2>/dev/null

# 运行（不使用 --skip_existing，会重新处理已完成的，但会覆盖）
nohup python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_val \
    --knn 5 \
    > tphate_run.log 2>&1 &

echo "作业已在后台运行，PID: $!"
echo "查看输出: tail -f tphate_run.log"
echo "检查进度: ls -1 aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l"






