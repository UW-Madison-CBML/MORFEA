#!/bin/bash
# 保存到 rho9 目录（有权限）

cd /staging/groups/bhaskar_group/rho9

echo "=== 使用 rho9 目录（有权限）==="
echo "输出目录: /staging/groups/bhaskar_group/rho9/v1_baseline_tphate"
echo ""

# 先创建输出目录（确保有权限）
mkdir -p /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots
mkdir -p /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/curvature_plots
echo "✓ 输出目录已创建"

echo ""
echo "运行脚本..."
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/rho9/v1_baseline_tphate \
    --knn 5 \
    > /tmp/tphate_run.log 2>&1 &

echo "✓ 作业已启动，PID: $!"
echo "日志: /tmp/tphate_run.log"
echo "查看日志: tail -f /tmp/tphate_run.log"
echo ""
echo "监控进度:"
echo "  while true; do"
echo "    COUNT=\$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l || echo 0)"
echo "    echo \"[\$(date '+%H:%M:%S')] \$COUNT / 704 plots\""
echo "    sleep 30"
echo "  done"






