#!/bin/bash
# 监控 ivf 目录的进度

echo "=== 检查进程状态 ==="
ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep

echo ""
echo "=== 监控进度 ==="
OUTPUT_DIR="/staging/groups/bhaskar_group/ivf/aadhitya_v1_val"

# 检查输出目录是否存在
if [ ! -d "$OUTPUT_DIR/tphate_plots" ]; then
    echo "输出目录不存在，可能还在初始化..."
    echo "等待 10 秒后再次检查..."
    sleep 10
fi

echo "开始监控（每30秒检查一次，按 Ctrl+C 停止）..."
echo ""

LAST_COUNT=0
ITERATION=0

while true; do
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l 2>/dev/null || echo "0")
    TIME=$(date '+%H:%M:%S')
    ITERATION=$((ITERATION + 1))
    
    if [ "$COUNT" -gt "$LAST_COUNT" ]; then
        INCREASED=$((COUNT - LAST_COUNT))
        echo "[$TIME] ✓ $COUNT plots (+$INCREASED) - 进程运行中"
        LAST_COUNT=$COUNT
    else
        # 每5次检查（约2.5分钟）显示一次状态
        if [ $((ITERATION % 5)) -eq 0 ]; then
            echo "[$TIME] $COUNT plots - 检查中..."
        fi
    fi
    
    # 检查进程是否还在运行
    if ! ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep > /dev/null; then
        echo ""
        echo "⚠️  进程已停止（可能完成或遇到错误）"
        echo "最终 plots: $COUNT"
        break
    fi
    
    # 如果达到目标（200个新胚胎），提示完成
    if [ "$COUNT" -ge 200 ]; then
        echo ""
        echo "✓ 第一批完成！已生成 $COUNT 个新 plots"
        break
    fi
    
    sleep 30
done

echo ""
echo "检查最后处理的胚胎:"
ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -5 | xargs -n1 basename | sed 's/_tphate\.png//'






