#!/bin/bash
# 检查可以删除的文件

echo "=== 检查配额 ==="
quota -s

echo ""
echo "=== 检查各个目录大小（按大小排序）==="
cd /staging/groups/bhaskar_group/rho9
du -sh * 2>/dev/null | sort -h

echo ""
echo "=== 检查日志文件 ==="
echo "日志文件："
find . -name "*.log" -type f -exec du -sh {} \; 2>/dev/null | sort -h

echo ""
echo "=== 检查临时文件 ==="
echo "tmp 目录："
du -sh tmp 2>/dev/null
ls -lh tmp/ 2>/dev/null | head -20

echo ""
echo "=== 检查旧的结果目录 ==="
echo "可能的旧结果目录："
ls -d */ 2>/dev/null | while read dir; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
        echo "$size  $dir"
    fi
done | sort -h

echo ""
echo "=== 检查 ~/logs 目录 ==="
du -sh ~/logs 2>/dev/null
ls -lh ~/logs/*.log 2>/dev/null | head -20






