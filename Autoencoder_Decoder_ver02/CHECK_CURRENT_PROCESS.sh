#!/bin/bash
# 检查当前进程使用的参数

echo "=== 检查当前运行的进程 ==="
ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep

echo ""
echo "=== 检查输出目录位置 ==="
echo "相对路径（可能在当前目录）:"
ls -d aadhitya_v1_val 2>/dev/null && echo "  ✓ 存在" || echo "  ✗ 不存在"

echo ""
echo "绝对路径（应该在 staging）:"
ls -d /staging/groups/bhaskar_group/rho9/aadhitya_v1_val 2>/dev/null && echo "  ✓ 存在" || echo "  ✗ 不存在"

echo ""
echo "=== 检查当前工作目录 ==="
pwd

echo ""
echo "如果使用相对路径，文件会保存在当前工作目录"
echo "应该使用绝对路径: --output_base /staging/groups/bhaskar_group/rho9/aadhitya_v1_val"






