#!/bin/bash
# 重命名脚本文件，去掉人名

cd /staging/groups/bhaskar_group/rho9

echo "=== 重命名脚本文件 ==="
if [ -f "generate_tphate_for_aadhitya.py" ]; then
    mv generate_tphate_for_aadhitya.py generate_tphate_plots.py
    echo "✓ 已重命名: generate_tphate_for_aadhitya.py -> generate_tphate_plots.py"
else
    echo "⚠️  文件不存在"
fi

echo ""
echo "=== 删除 aadhitya_v1_test 目录 ==="
if [ -d "aadhitya_v1_test" ]; then
    rm -rf aadhitya_v1_test
    echo "✓ 已删除 aadhitya_v1_test"
else
    echo "目录不存在"
fi

echo ""
echo "=== 确认清理 ==="
find . -iname "*aadhitya*" 2>/dev/null

echo ""
echo "=== 检查配额 ==="
quota -s






