#!/bin/bash
# 检查磁盘配额和使用情况

echo "=== 检查 staging 目录配额 ==="
quota -s /staging/groups/bhaskar_group/rho9 2>/dev/null || echo "无法查询配额（可能需要特定命令）"

echo ""
echo "=== 检查磁盘使用情况 ==="
df -h /staging/groups/bhaskar_group/rho9

echo ""
echo "=== 检查输出目录大小 ==="
du -sh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val 2>/dev/null
du -sh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots 2>/dev/null
du -sh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/curvature_plots 2>/dev/null

echo ""
echo "=== 计算单个 plot 的平均大小 ==="
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
if [ $COUNT -gt 0 ]; then
    TOTAL_SIZE=$(du -sb /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots 2>/dev/null | awk '{print $1}')
    AVG_SIZE=$((TOTAL_SIZE / COUNT))
    AVG_SIZE_MB=$((AVG_SIZE / 1024 / 1024))
    echo "已生成: $COUNT 个 plots"
    echo "总大小: $(du -sh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots 2>/dev/null | awk '{print $1}')"
    echo "平均每个: ${AVG_SIZE_MB} MB"
    echo ""
    ESTIMATED_TOTAL=$((AVG_SIZE_MB * 704))
    echo "估计 704 个胚胎总大小: ${ESTIMATED_TOTAL} MB ($(($ESTIMATED_TOTAL / 1024)) GB)"
fi

echo ""
echo "=== 检查是否有其他位置可以保存 ==="
echo "可能的解决方案："
echo "  1. 降低 DPI（从 300 降到 150 或 100）"
echo "  2. 使用压缩（JPEG 格式而不是 PNG）"
echo "  3. 联系管理员增加配额"
echo "  4. 使用其他存储位置"
