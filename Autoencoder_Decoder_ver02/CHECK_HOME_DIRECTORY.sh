#!/bin/bash
# 检查 home 目录，找出可以删除的内容

echo "=== Home 目录总大小 ==="
du -sh ~

echo ""
echo "=== Home 目录下各个目录大小（按大小排序）==="
du -sh ~/* ~/.[^.]* 2>/dev/null | sort -hr | head -20

echo ""
echo "=== 检查常见的大目录 ==="
for dir in ~/.cache ~/.local ~/logs ~/tmp ~/.conda ~/.ipython; do
    if [ -d "$dir" ]; then
        echo "$dir:"
        du -sh "$dir"
    fi
done

echo ""
echo "=== 检查所有大于 10M 的文件 ==="
find ~ -type f -size +10M -exec du -sh {} \; 2>/dev/null | sort -hr | head -20

echo ""
echo "=== 检查日志文件 ==="
find ~ -type f -name "*.log" -size +1M -exec du -sh {} \; 2>/dev/null | sort -hr | head -10

echo ""
echo "=== 当前配额 ==="
quota -s






