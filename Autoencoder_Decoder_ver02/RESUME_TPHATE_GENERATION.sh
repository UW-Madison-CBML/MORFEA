#!/bin/bash
# 在CHTC上执行：恢复T-PHATE生成

echo "============================================================"
echo "恢复 T-PHATE 生成 - 清理空间并继续"
echo "============================================================"
echo ""

# 1. 检查当前状态
OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/v1_baseline_tphate"
COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l || echo "0")
echo "当前进度: $COUNT / 704 plots"
echo ""

# 2. 检查磁盘使用
echo "磁盘使用情况："
df -h /staging/groups/bhaskar_group/rho9 | grep -E "(Filesystem|/dev/md9)"
echo ""

# 3. 停止当前进程
echo "停止当前进程..."
pkill -f "generate_tphate_plots.py" 2>/dev/null
pkill -f "generate_tphate_for_aadhitya.py" 2>/dev/null
sleep 2
echo "✓ 已停止"
echo ""

# 4. 检查并删除旧的 aadhitya_v1_val 目录（如果存在）
if [ -d "/staging/groups/bhaskar_group/rho9/aadhitya_v1_val" ]; then
    OLD_SIZE=$(du -sh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val 2>/dev/null | cut -f1)
    echo "发现旧目录: aadhitya_v1_val (大小: $OLD_SIZE)"
    echo "删除旧目录以释放空间..."
    rm -rf /staging/groups/bhaskar_group/rho9/aadhitya_v1_val
    echo "✓ 已删除"
    echo ""
fi

# 5. 再次检查磁盘使用
echo "清理后的磁盘使用情况："
df -h /staging/groups/bhaskar_group/rho9 | grep -E "(Filesystem|/dev/md9)"
echo ""

# 6. 恢复处理（使用 --skip_existing）
echo "恢复处理（跳过已存在的plots）..."
echo ""
nohup python3 /staging/groups/bhaskar_group/rho9/generate_tphate_plots.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/rho9/v1_baseline_tphate \
    --knn 5 \
    --skip_existing \
    > /tmp/tphate_run.log 2>&1 &

PID=$!
echo "✓ 作业已启动，PID: $PID"
echo ""
echo "查看日志："
echo "  tail -f /tmp/tphate_run.log"
echo ""
echo "监控进度："
echo "  while true; do COUNT=\$(ls -1 $OUTPUT_DIR/tphate_plots/*.png 2>/dev/null | wc -l); echo \"\$(date '+%H:%M:%S'): \$COUNT / 704\"; sleep 30; done"
echo ""
echo "============================================================"






