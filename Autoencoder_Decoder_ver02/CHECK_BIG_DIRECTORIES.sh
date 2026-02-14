#!/bin/bash
# 检查大目录，找出真正占用空间的文件

cd /staging/groups/bhaskar_group/rho9

echo "=== 检查各个大目录 ==="
du -sh * 2>/dev/null | sort -hr | head -10

echo ""
echo "=== 检查 checkpoints 目录内容 ==="
if [ -d "checkpoints" ]; then
    echo "checkpoints 目录大小: $(du -sh checkpoints | awk '{print $1}')"
    echo "内容："
    ls -lh checkpoints/ | head -20
    echo ""
    echo "如果有多个 checkpoint，可以考虑只保留最新的"
fi

echo ""
echo "=== 检查是否有其他大文件 ==="
find . -type f -size +10M -exec du -sh {} \; 2>/dev/null | sort -hr






