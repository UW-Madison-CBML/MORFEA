#!/bin/bash
# 检查进程并监控进度（不使用日志文件）

echo "=== 检查进程状态 ==="
ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep

echo ""
echo "=== 检查输出目录 ==="
OUTPUT_DIR="/staging/groups/bhaskar_group/ivf/aadhitya_v1_val"
if [ -d "$OUTPUT_DIR/tphate_plots" ]; then
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    echo "✓ 输出目录存在"
    echo "当前 plots: $COUNT"
    
    if [ $COUNT -gt 0 ]; then
        echo ""
        echo "最后处理的胚胎:"
        ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -5 | xargs -n1 basename | sed 's/_tphate\.png//'
    fi
else
    echo "⚠️  输出目录不存在（可能还在初始化）"
fi

echo ""
echo "=== 监控进度（每30秒检查一次）==="
echo "按 Ctrl+C 停止"
echo ""

LAST_COUNT=0
while true; do
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    TIME=$(date '+%H:%M:%S')
    
    if [ $COUNT -gt $LAST_COUNT ]; then
        INCREASED=$((COUNT - LAST_COUNT))
        echo "[$TIME] ✓ $COUNT plots (+$INCREASED)"
        LAST_COUNT=$COUNT
    else
        echo "[$TIME] $COUNT plots (无变化)"
    fi
    
    # 检查进程是否还在运行
    if ! ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep > /dev/null; then
        echo ""
        echo "⚠️  进程已停止"
        break
    fi
    
    sleep 30
done






