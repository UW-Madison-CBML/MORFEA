#!/bin/bash
# 检查 T-PHATE 生成作业的进度

echo "=== 检查作业进度 ==="
echo ""

# 1. 检查作业状态
echo "1. 作业状态:"
condor_q -submitter rho9 | grep -E "(ID|generate_tphate|RUN|IDLE|HOLD)"
echo ""

# 2. 检查已生成的 plot 数量
OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
if [ -d "$OUTPUT_DIR/tphate_plots" ]; then
    TPHATE_COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    CURVATURE_COUNT=$(ls -1 "$OUTPUT_DIR/curvature_plots"/*.png 2>/dev/null | wc -l)
    echo "2. 已生成的 plots:"
    echo "   T-PHATE plots: $TPHATE_COUNT / 704"
    echo "   Curvature plots: $CURVATURE_COUNT / 704"
    echo "   进度: $(( TPHATE_COUNT * 100 / 704 ))%"
    echo ""
    
    if [ $TPHATE_COUNT -gt 0 ]; then
        echo "   最后处理的胚胎:"
        ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -3 | sed 's/.*\///; s/_tphate\.png//'
    fi
else
    echo "2. 输出目录不存在或为空"
fi

echo ""

# 3. 检查日志文件
echo "3. 日志文件:"
LATEST_LOG=$(ls -t ~/logs/generate_tphate_*.out 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "   最新输出日志: $LATEST_LOG"
    echo "   文件大小: $(ls -lh "$LATEST_LOG" | awk '{print $5}')"
    echo ""
    echo "   最后 10 行输出:"
    tail -10 "$LATEST_LOG"
else
    echo "   ⚠️  没有找到日志文件"
fi

echo ""
echo "=== 查看实时日志 ==="
echo "运行: tail -f ~/logs/generate_tphate_*.out"






