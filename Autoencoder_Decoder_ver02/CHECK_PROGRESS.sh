#!/bin/bash
# 检查当前T-PHATE生成进度

echo "============================================================"
echo "检查 T-PHATE 生成进度"
echo "============================================================"
echo ""
echo "在 CHTC 上执行以下命令："
echo ""
cat << 'EOF'
# 1. 检查已生成的图片数量
OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/v1_baseline_tphate"
T_PHATE_COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l || echo "0")
CURVATURE_COUNT=$(ls -1 "$OUTPUT_DIR/curvature_plots"/*.png 2>/dev/null | wc -l || echo "0")

echo "T-PHATE plots: $T_PHATE_COUNT / 704"
echo "Curvature plots: $CURVATURE_COUNT / 704"
echo "总进度: $T_PHATE_COUNT / 704 ($(echo "scale=1; $T_PHATE_COUNT*100/704" | bc)%)"
echo ""

# 2. 检查磁盘使用
echo "磁盘使用情况："
df -h /staging/groups/bhaskar_group/rho9 | grep -E "(Filesystem|/dev/md9)"
echo ""

# 3. 检查进程是否在运行
echo "进程状态："
if ps aux | grep "generate_tphate" | grep -v grep > /dev/null; then
    echo "✓ 进程正在运行"
    ps aux | grep "generate_tphate" | grep -v grep | head -1
else
    echo "✗ 进程未运行"
fi
echo ""

# 4. 查看最后处理的几个胚胎
echo "最后处理的胚胎（按时间排序）："
ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -5 | xargs -n1 basename | sed 's/_tphate.png//'
EOF

echo ""
echo "============================================================"
