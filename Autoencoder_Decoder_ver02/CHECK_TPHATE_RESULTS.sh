#!/bin/bash
# 检查生成的 T-PHATE 和 Curvature plots

echo "=== 检查生成的 Plot 文件 ==="
echo ""

BASE_DIR="aadhitya_v1_test"

if [ -d "$BASE_DIR" ]; then
    echo "✓ 输出目录存在: $BASE_DIR"
    echo ""
    
    # 检查 T-PHATE plots
    if [ -d "$BASE_DIR/tphate_plots" ]; then
        echo "T-PHATE plots:"
        ls -lh "$BASE_DIR/tphate_plots/"*.png 2>/dev/null | head -10
        COUNT=$(ls -1 "$BASE_DIR/tphate_plots/"*.png 2>/dev/null | wc -l)
        echo "  总计: $COUNT 个文件"
        echo ""
    else
        echo "⚠️  T-PHATE plots 目录不存在"
    fi
    
    # 检查 Curvature plots
    if [ -d "$BASE_DIR/curvature_plots" ]; then
        echo "Curvature plots:"
        ls -lh "$BASE_DIR/curvature_plots/"*.png 2>/dev/null | head -10
        COUNT=$(ls -1 "$BASE_DIR/curvature_plots/"*.png 2>/dev/null | wc -l)
        echo "  总计: $COUNT 个文件"
        echo ""
    else
        echo "⚠️  Curvature plots 目录不存在"
    fi
    
    # 检查文件大小
    echo "文件大小统计:"
    du -sh "$BASE_DIR/tphate_plots" 2>/dev/null
    du -sh "$BASE_DIR/curvature_plots" 2>/dev/null
    
else
    echo "❌ 输出目录不存在: $BASE_DIR"
fi






