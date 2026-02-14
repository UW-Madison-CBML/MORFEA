#!/bin/bash
# 检查实际的磁盘配额和使用情况

echo "=== 检查配额 ==="
quota -s 2>/dev/null || echo "无法查询配额"

echo ""
echo "=== 检查磁盘使用情况 ==="
df -h /staging/groups/bhaskar_group/rho9
df -h /staging/groups/bhaskar_group/ivf

echo ""
echo "=== 检查输出目录大小 ==="
if [ -d "/staging/groups/bhaskar_group/rho9/v1_baseline_tphate" ]; then
    du -sh /staging/groups/bhaskar_group/rho9/v1_baseline_tphate
    du -sh /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots
    du -sh /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/curvature_plots
    
    echo ""
    echo "=== 计算平均文件大小 ==="
    COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
    if [ $COUNT -gt 0 ]; then
        TOTAL_SIZE=$(du -sb /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots 2>/dev/null | awk '{print $1}')
        AVG_SIZE_MB=$((TOTAL_SIZE / COUNT / 1024 / 1024))
        echo "已生成: $COUNT 个 T-PHATE plots"
        echo "总大小: $(du -sh /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots 2>/dev/null | awk '{print $1}')"
        echo "平均每个: ${AVG_SIZE_MB} MB"
        
        ESTIMATED=$((AVG_SIZE_MB * 704 * 2))  # *2 因为还有 curvature plots
        echo "估计总大小 (704 个胚胎，两种 plot): ${ESTIMATED} MB ($(($ESTIMATED / 1024)) GB)"
    fi
fi

echo ""
echo "=== 检查 rho9 目录总大小 ==="
du -sh /staging/groups/bhaskar_group/rho9 2>/dev/null | head -1






