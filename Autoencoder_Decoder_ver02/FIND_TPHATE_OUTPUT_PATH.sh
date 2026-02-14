#!/bin/bash
# 查找 T-PHATE 输出文件的完整路径

echo "=== 查找 T-PHATE 输出文件路径 ==="
echo ""

# 当前目录
CURRENT_DIR=$(pwd)
echo "当前目录: $CURRENT_DIR"
echo ""

# 查找输出目录
OUTPUT_DIRS=(
    "aadhitya_v1_test"
    "/staging/groups/bhaskar_group/rho9/aadhitya_v1_test"
    "$HOME/aadhitya_v1_test"
)

for dir in "${OUTPUT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ 找到输出目录: $dir"
        echo "  完整路径: $(realpath "$dir" 2>/dev/null || echo "$dir")"
        echo ""
        
        # 列出内容
        echo "  目录内容:"
        if [ -d "$dir/tphate_plots" ]; then
            echo "    - tphate_plots/ ($(ls -1 "$dir/tphate_plots"/*.png 2>/dev/null | wc -l) 个文件)"
        fi
        if [ -d "$dir/curvature_plots" ]; then
            echo "    - curvature_plots/ ($(ls -1 "$dir/curvature_plots"/*.png 2>/dev/null | wc -l) 个文件)"
        fi
        echo ""
    fi
done






