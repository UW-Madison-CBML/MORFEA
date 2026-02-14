#!/bin/bash
# 查找所有包含 aadhitya 的文件和目录

cd /staging/groups/bhaskar_group/rho9

echo "=== 查找所有包含 aadhitya 的文件和目录 ==="
find . -iname "*aadhitya*" -type f 2>/dev/null
find . -iname "*aadhitya*" -type d 2>/dev/null

echo ""
echo "=== 检查输出目录 ==="
ls -d */ 2>/dev/null | grep -i aadhitya

echo ""
echo "=== 重命名建议 ==="
echo "如果当前输出目录是 v1_baseline_tphate，应该没问题"
echo "如果有 aadhitya_v1_test，可以删除或重命名"






