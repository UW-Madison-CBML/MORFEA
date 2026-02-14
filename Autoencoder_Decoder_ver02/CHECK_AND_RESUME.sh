#!/bin/bash
# 检查状态并从断点继续

cd /staging/groups/bhaskar_group/rho9

echo "=== 检查当前进度 ==="
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
echo "已完成: $COUNT / 704 plots"
echo ""

echo "=== 检查进程状态 ==="
if ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep > /dev/null; then
    echo "✓ 进程仍在运行"
    ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep
else
    echo "⚠️  进程已停止"
    echo ""
    echo "=== 从断点继续运行 ==="
    echo "从第 $COUNT 个胚胎开始继续..."
    echo ""
    
    python3 generate_tphate_for_aadhitya.py \
        --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
        --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
        --output_base /staging/groups/bhaskar_group/rho9/v1_baseline_tphate \
        --knn 5 \
        --start_from $COUNT \
        > /tmp/tphate_run.log 2>&1 &
    
    echo "✓ 已重新启动，PID: $!"
    echo "查看日志: tail -f /tmp/tphate_run.log"
fi






