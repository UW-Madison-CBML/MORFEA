#!/bin/bash
# 快速检查 ivf 目录的进度

OUTPUT_DIR="/staging/groups/bhaskar_group/ivf/aadhitya_v1_val"

echo "=== 检查进程 ==="
if ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep > /dev/null; then
    echo "✓ 进程正在运行"
    ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep | awk '{print "  PID: "$2", CPU: "$3"%"}'
else
    echo "⚠️  进程不在运行"
fi

echo ""
echo "=== 检查输出目录 ==="
if [ -d "$OUTPUT_DIR/tphate_plots" ]; then
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    echo "✓ 输出目录存在"
    echo "当前 plots: $COUNT"
    
    if [ $COUNT -gt 0 ]; then
        echo ""
        echo "最后处理的胚胎:"
        ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -3 | xargs -n1 basename | sed 's/_tphate\.png//'
    else
        echo "还没有生成 plots（可能还在初始化）"
    fi
else
    echo "⚠️  输出目录不存在（可能还在初始化）"
fi






