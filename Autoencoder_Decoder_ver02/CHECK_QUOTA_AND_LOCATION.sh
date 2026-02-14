#!/bin/bash
# 检查配额和保存位置

echo "=== 检查当前保存位置 ==="
echo "当前输出目录: /staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
du -sh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val 2>/dev/null

echo ""
echo "=== 检查 ivf 目录 ==="
echo "ivf 目录: /staging/groups/bhaskar_group/ivf"
ls -ld /staging/groups/bhaskar_group/ivf 2>/dev/null
du -sh /staging/groups/bhaskar_group/ivf 2>/dev/null

echo ""
echo "=== 检查 rho9 目录配额（如果可查询）==="
quota -s 2>/dev/null || df -h /staging/groups/bhaskar_group/rho9

echo ""
echo "=== 建议 ==="
echo "如果 rho9 目录有配额限制，可以考虑："
echo "  1. 保存到 ivf 目录: /staging/groups/bhaskar_group/ivf/aadhitya_v1_val"
echo "  2. 或者联系管理员增加 rho9 目录配额"






