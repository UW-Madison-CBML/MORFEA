#!/bin/bash
# 快速检查配额和大小

echo "=== 检查输出目录大小 ==="
du -sh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val 2>/dev/null

echo ""
echo "=== 计算平均大小 ==="
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
if [ $COUNT -gt 0 ]; then
    TOTAL_SIZE=$(du -sb /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots 2>/dev/null | awk '{print $1}')
    AVG_SIZE_MB=$((TOTAL_SIZE / COUNT / 1024 / 1024))
    echo "已生成: $COUNT 个 plots"
    echo "平均每个: ${AVG_SIZE_MB} MB"
    ESTIMATED=$((AVG_SIZE_MB * 704))
    echo "估计总大小: ${ESTIMATED} MB ($(($ESTIMATED / 1024)) GB)"
fi






