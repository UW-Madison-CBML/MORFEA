#!/bin/bash
# Check rho9/ivf_data directory structure

echo "=== 检查 /staging/groups/bhaskar_group/rho9/ivf_data/ ==="
echo ""

DATA_DIR="/staging/groups/bhaskar_group/rho9/ivf_data"

if [ -d "$DATA_DIR" ]; then
    echo "目录内容："
    ls -lh "$DATA_DIR"
    echo ""
    
    # 检查是否有 embryo_dataset 目录
    if [ -d "$DATA_DIR/embryo_dataset" ]; then
        echo "✓ 找到 embryo_dataset 目录"
        echo "  路径: $DATA_DIR/embryo_dataset"
        echo ""
        echo "检查子目录（前5个）："
        ls -d "$DATA_DIR/embryo_dataset"/*/ 2>/dev/null | head -5
        echo ""
        echo "检查是否有图像文件："
        find "$DATA_DIR/embryo_dataset" -name "*.jpeg" -type f 2>/dev/null | head -3
    else
        echo "⚠️  没有找到 embryo_dataset 子目录"
        echo ""
        echo "检查是否有 tar.gz 文件："
        ls -lh "$DATA_DIR"/*.tar.gz 2>/dev/null
    fi
else
    echo "❌ 目录不存在"
fi

