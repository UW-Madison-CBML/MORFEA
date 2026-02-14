#!/bin/bash
# 详细检查所有可以删除的内容

cd /staging/groups/bhaskar_group/rho9

echo "=== 所有目录和文件大小（按大小排序）==="
du -sh * .[^.]* 2>/dev/null | sort -hr

echo ""
echo "=== 详细检查每个目录 ==="
echo ""
echo "1. v1_baseline_tphate 目录:"
du -sh v1_baseline_tphate/*
ls -lh v1_baseline_tphate/tphate_plots/ | head -5
ls -lh v1_baseline_tphate/curvature_plots/ | head -5

echo ""
echo "2. checkpoints 目录:"
ls -lh checkpoints/

echo ""
echo "3. 其他目录内容:"
for dir in */; do
    if [ "$dir" != "v1_baseline_tphate/" ] && [ "$dir" != "checkpoints/" ]; then
        echo "  $dir:"
        ls -lh "$dir" 2>/dev/null | head -10
        echo ""
    fi
done

echo ""
echo "=== 查找所有大于 1M 的文件 ==="
find . -type f -size +1M -exec du -sh {} \; 2>/dev/null | sort -hr

echo ""
echo "=== 查找压缩文件 ==="
find . -type f \( -name "*.tar.gz" -o -name "*.zip" -o -name "*.tar" -o -name "*.gz" \) -exec du -sh {} \; 2>/dev/null | sort -hr

echo ""
echo "=== 查找日志文件 ==="
find . -type f -name "*.log" -exec du -sh {} \; 2>/dev/null | sort -hr

echo ""
echo "=== 检查临时目录 ==="
if [ -d "tmp" ]; then
    echo "tmp 目录内容:"
    ls -lha tmp/ | head -20
fi

echo ""
echo "=== 当前配额 ==="
quota -s






