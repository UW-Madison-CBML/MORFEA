#!/bin/bash
# 保存到 ivf 目录（可能没有配额限制）

cd /staging/groups/bhaskar_group/rho9

echo "=== 检查 ivf 目录权限 ==="
ls -ld /staging/groups/bhaskar_group/ivf/

echo ""
echo "=== 尝试在 ivf 目录创建输出目录 ==="
mkdir -p /staging/groups/bhaskar_group/ivf/v1_baseline_tphate/tphate_plots
mkdir -p /staging/groups/bhaskar_group/ivf/v1_baseline_tphate/curvature_plots

if [ $? -eq 0 ]; then
    echo "✓ 成功创建目录！可以在 ivf 目录保存"
    echo ""
    echo "=== 继续运行（保存到 ivf 目录）==="
    
    # 检查当前进度（从 rho9 目录）
    COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
    echo "当前进度: $COUNT / 704"
    echo ""
    echo "从 $COUNT 开始继续，保存到 ivf 目录："
    
    python3 generate_tphate_plots.py \
        --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
        --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
        --output_base /staging/groups/bhaskar_group/ivf/v1_baseline_tphate \
        --knn 5 \
        --start_from $COUNT \
        > /tmp/tphate_run.log 2>&1 &
    
    echo "✓ 作业已启动，PID: $!"
    echo "输出目录: /staging/groups/bhaskar_group/ivf/v1_baseline_tphate"
    echo "查看日志: tail -f /tmp/tphate_run.log"
else
    echo "⚠️  无法创建目录，权限不足"
fi






