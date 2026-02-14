#!/bin/bash
# 检查整个目录，找出大文件和可以删除的内容

cd /staging/groups/bhaskar_group/rho9

echo "=== 检查所有目录大小（按大小排序）==="
du -sh * 2>/dev/null | sort -hr

echo ""
echo "=== 检查所有大于 10M 的文件 ==="
find . -type f -size +10M -exec du -sh {} \; 2>/dev/null | sort -hr

echo ""
echo "=== 检查 checkpoints 目录内容 ==="
if [ -d "checkpoints" ]; then
    ls -lh checkpoints/
    echo ""
    echo "如果有多份 checkpoint，可以只保留最新的"
fi

echo ""
echo "=== 检查是否有重复或旧的目录 ==="
ls -d */ 2>/dev/null

echo ""
echo "=== 检查是否有压缩文件（.tar.gz, .zip等）==="
find . -type f \( -name "*.tar.gz" -o -name "*.zip" -o -name "*.tar" \) -exec du -sh {} \; 2>/dev/null | sort -hr

echo ""
echo "=== 检查当前配额 ==="
quota -s






