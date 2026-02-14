#!/bin/bash
# 检查配额并给出建议

echo "=== 当前配额使用 ==="
quota -s 2>/dev/null

echo ""
echo "=== 输出目录大小 ==="
du -sh /staging/groups/bhaskar_group/rho9/v1_baseline_tphate

echo ""
echo "=== 当前进度 ==="
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
echo "已完成: $COUNT / 704 个胚胎"

echo ""
echo "=== 剩余空间估算 ==="
QUOTA_USED=$(quota -s 2>/dev/null | grep "/dev/md9" | awk '{print $2}' | sed 's/M//')
QUOTA_LIMIT=$(quota -s 2>/dev/null | grep "/dev/md9" | awk '{print $3}' | sed 's/M//')
AVAILABLE=$((QUOTA_LIMIT - QUOTA_USED))
echo "可用空间: ${AVAILABLE}M"

CURRENT_SIZE=$(du -sm /staging/groups/bhaskar_group/rho9/v1_baseline_tphate 2>/dev/null | awk '{print $1}')
AVG_PER_EMBRYO=$((CURRENT_SIZE * 1024 / COUNT))  # MB per embryo (2 plots)
REMAINING_EMBRYOS=$((704 - COUNT))
ESTIMATED_NEEDED=$((AVG_PER_EMBRYO * REMAINING_EMBRYOS / 1024))
echo "当前平均: ${AVG_PER_EMBRYO} MB/胚胎"
echo "估计还需要: ${ESTIMATED_NEEDED}M"

echo ""
if [ $ESTIMATED_NEEDED -gt $AVAILABLE ]; then
    echo "⚠️  估计需要的空间 (${ESTIMATED_NEEDED}M) > 可用空间 (${AVAILABLE}M)"
    echo "建议："
    echo "  1. 只处理 validation set"
    echo "  2. 联系管理员增加配额"
    echo "  3. 压缩已完成的 plots 释放空间"
else
    echo "✓ 可用空间应该足够"
fi






