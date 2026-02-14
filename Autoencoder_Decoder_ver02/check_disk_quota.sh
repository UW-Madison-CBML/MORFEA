#!/bin/bash


echo ""
df -h /staging/groups/bhaskar_group/rho9

echo ""
du -sh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val 2>/dev/null
du -sh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots 2>/dev/null
du -sh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/curvature_plots 2>/dev/null

echo ""
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
if [ $COUNT -gt 0 ]; then
    TOTAL_SIZE=$(du -sb /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots 2>/dev/null | awk '{print $1}')
    AVG_SIZE=$((TOTAL_SIZE / COUNT))
    AVG_SIZE_MB=$((AVG_SIZE / 1024 / 1024))
    echo ""
    ESTIMATED_TOTAL=$((AVG_SIZE_MB * 704))
fi

echo ""
