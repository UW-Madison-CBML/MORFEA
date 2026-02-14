#!/bin/bash

du -sh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val 2>/dev/null

echo ""
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
if [ $COUNT -gt 0 ]; then
    TOTAL_SIZE=$(du -sb /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots 2>/dev/null | awk '{print $1}')
    AVG_SIZE_MB=$((TOTAL_SIZE / COUNT / 1024 / 1024))
    ESTIMATED=$((AVG_SIZE_MB * 704))
fi






