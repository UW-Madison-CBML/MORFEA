#!/bin/bash
# 检查 ivf 目录的权限

echo "=== ivf 目录权限 ==="
ls -ld /staging/groups/bhaskar_group/ivf/

echo ""
echo "=== 当前用户和组 ==="
id

echo ""
echo "=== ivf 目录的所有文件和子目录 ==="
ls -la /staging/groups/bhaskar_group/ivf/ | head -20

echo ""
echo "=== 检查是否有现有子目录可以写入 ==="
# 检查 latents 目录的权限
ls -ld /staging/groups/bhaskar_group/ivf/latents/

echo ""
echo "=== 尝试在现有子目录中创建文件 ==="
# 尝试在 latents 目录中创建测试文件
TEST_FILE="/staging/groups/bhaskar_group/ivf/latents/test_$(date +%s)"
touch "$TEST_FILE" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ 可以在 latents 子目录创建文件"
    rm "$TEST_FILE"
else
    echo "✗ 无法在 latents 子目录创建文件"
    touch "$TEST_FILE" 2>&1
fi






