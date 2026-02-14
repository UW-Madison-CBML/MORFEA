#!/bin/bash
# Find the correct data path on CHTC

echo "=== 查找数据目录 ==="
echo ""

# 检查所有可能的数据路径
POSSIBLE_PATHS=(
    "/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset"
    "/staging/groups/bhaskar_group/ivf/embryo_dataset"
    "/staging/groups/bhaskar_group/ivf"
    "/staging/groups/bhaskar_group/rho9/ivf_data"
    "/project/bhaskar_group/ivf/embryo_dataset"
    "/project/bhaskar_group/ivf"
)

echo "检查可能的路径："
FOUND=false
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "  ✓ 找到: $path"
        # 检查是否包含 embryo_dataset 目录
        if [ -d "$path/embryo_dataset" ]; then
            echo "    包含 embryo_dataset 子目录"
            FOUND=true
            DATA_PATH="$path/embryo_dataset"
        elif [ "$(basename $path)" = "embryo_dataset" ]; then
            echo "    这就是 embryo_dataset 目录"
            FOUND=true
            DATA_PATH="$path"
        else
            # 检查是否有子目录看起来像数据
            echo "    检查子目录..."
            ls -d "$path"/*/ 2>/dev/null | head -3 | while read dir; do
                if [ -d "$dir" ] && [ -n "$(ls -A "$dir"/*.jpeg 2>/dev/null | head -1)" ]; then
                    echo "      可能的数据目录: $dir"
                fi
            done
        fi
    else
        echo "  ✗ 不存在: $path"
    fi
done

echo ""
if [ "$FOUND" = true ]; then
    echo "=== 找到数据目录: $DATA_PATH ==="
    echo ""
    echo "创建 data symlink："
    echo "  ln -sf $DATA_PATH data"
    echo ""
    echo "或者直接使用："
    echo "  python3 build_index.py --root $DATA_PATH --out index.csv"
else
    echo "=== 未找到数据目录 ==="
    echo ""
    echo "请手动检查 staging 路径："
    echo "  ls -la /staging/groups/bhaskar_group/"
    echo "  ls -la /staging/groups/bhaskar_group/rho9/"
    echo "  ls -la /staging/groups/bhaskar_group/ivf/"
fi
