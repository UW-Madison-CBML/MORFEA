#!/bin/bash
# 清理 home 目录并继续运行

cd /staging/groups/bhaskar_group/rho9

echo "=== 清理 home 目录 ==="

# 1. 删除 .cache（最安全）
echo "删除 .cache..."
rm -rf ~/.cache
echo "✓ 已删除 .cache"

# 2. 删除压缩文件（如果不需要）
if [ -f ~/ivf-embryo-analysis-Raffael.tgz ]; then
    echo "删除 ivf-embryo-analysis-Raffael.tgz..."
    rm ~/ivf-embryo-analysis-Raffael.tgz
    echo "✓ 已删除压缩文件"
fi

echo ""
echo "=== 检查配额 ==="
quota -s

echo ""
echo "=== 检查当前进度 ==="
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
echo "已完成: $COUNT / 704 个胚胎"

echo ""
echo "=== 继续运行 ==="
python3 generate_tphate_plots.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/rho9/v1_baseline_tphate \
    --knn 5 \
    --start_from $COUNT \
    > /tmp/tphate_run.log 2>&1 &

echo "✓ 作业已启动，PID: $!"
echo "查看日志: tail -f /tmp/tphate_run.log"
echo ""
echo "监控进度:"
echo "  while true; do"
echo "    COUNT=\$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)"
echo "    echo \"[\$(date '+%H:%M:%S')] \$COUNT / 704 plots\""
echo "    sleep 30"
echo "  done"






