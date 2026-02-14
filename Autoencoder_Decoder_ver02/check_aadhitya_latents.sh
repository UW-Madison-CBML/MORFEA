#!/bin/bash
# 查找 Aadhitya 的 latent 文件

echo "=== 查找 Aadhitya 的 Latent 文件 ==="
echo ""

# 可能的路径
POSSIBLE_PATHS=(
    "/staging/groups/bhaskar_group/ivf"
    "/staging/groups/bhaskar_group/ivf/data"
    "/staging/groups/bhaskar_group/ivf/latents"
    "/staging/groups/bhaskar_group/rho9/ivf"
    "/staging/groups/bhaskar_group/rho9/ivf/data"
)

echo "1. 检查常见路径："
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "  ✓ 目录存在: $path"
        if [ -f "$path/latents.npy" ]; then
            echo "    ✓ 找到 latents.npy"
            ls -lh "$path/latents.npy"
        fi
        if [ -f "$path/latents.csv" ]; then
            echo "    ✓ 找到 latents.csv"
            ls -lh "$path/latents.csv"
        fi
        echo "    目录内容："
        ls -lh "$path" | head -10
        echo ""
    fi
done

echo ""
echo "2. 搜索所有 latents.npy 文件："
find /staging/groups/bhaskar_group -name "latents.npy" -type f 2>/dev/null | head -10

echo ""
echo "3. 搜索所有 latents.csv 文件："
find /staging/groups/bhaskar_group -name "latents.csv" -type f 2>/dev/null | head -10

echo ""
echo "4. 检查 ivf 相关目录："
find /staging/groups/bhaskar_group -type d -name "*ivf*" 2>/dev/null

echo ""
echo "=== 完成 ==="






