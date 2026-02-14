#!/bin/bash


echo ""
df -h /staging/groups/bhaskar_group/rho9
df -h /staging/groups/bhaskar_group/ivf

echo ""
if [ -d "/staging/groups/bhaskar_group/rho9/v1_baseline_tphate" ]; then
    du -sh /staging/groups/bhaskar_group/rho9/v1_baseline_tphate
    du -sh /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots
    du -sh /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/curvature_plots
    
    echo ""
    COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
    if [ $COUNT -gt 0 ]; then
        TOTAL_SIZE=$(du -sb /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots 2>/dev/null | awk '{print $1}')
        AVG_SIZE_MB=$((TOTAL_SIZE / COUNT / 1024 / 1024))
        
        ESTIMATED=$((AVG_SIZE_MB * 704 * 2))
    fi
fi

echo ""
du -sh /staging/groups/bhaskar_group/rho9 2>/dev/null | head -1






