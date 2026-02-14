#!/bin/bash
# 详细检查 rho9 目录

cd /staging/groups/bhaskar_group/rho9

echo "=== rho9 目录总大小 ==="
du -sh .

echo ""
echo "=== 所有目录大小（按大小排序）==="
du -sh * 2>/dev/null | sort -hr

echo ""
echo "=== 所有大于 1M 的文件 ==="
find . -type f -size +1M -exec du -sh {} \; 2>/dev/null | sort -hr

echo ""
echo "=== 检查子目录详情 ==="
echo "v1_baseline_tphate:"
du -sh v1_baseline_tphate/* 2>/dev/null | sort -hr

echo ""
echo "checkpoints:"
ls -lh checkpoints/

echo ""
echo "其他目录:"
ls -d */ 2>/dev/null | grep -v "v1_baseline_tphate\|checkpoints"

echo ""
echo "=== 检查是否有隐藏文件或目录 ==="
du -sh .[^.]* 2>/dev/null | sort -hr

echo ""
echo "=== 当前配额 ==="
quota -s






