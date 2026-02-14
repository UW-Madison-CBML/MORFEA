#!/bin/bash

quota -s 2>/dev/null

echo ""
du -sh /staging/groups/bhaskar_group/rho9/v1_baseline_tphate

echo ""
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)

echo ""
QUOTA_USED=$(quota -s 2>/dev/null | grep "/dev/md9" | awk '{print $2}' | sed 's/M//')
QUOTA_LIMIT=$(quota -s 2>/dev/null | grep "/dev/md9" | awk '{print $3}' | sed 's/M//')
AVAILABLE=$((QUOTA_LIMIT - QUOTA_USED))

CURRENT_SIZE=$(du -sm /staging/groups/bhaskar_group/rho9/v1_baseline_tphate 2>/dev/null | awk '{print $1}')
AVG_PER_EMBRYO=$((CURRENT_SIZE * 1024 / COUNT))  # MB per embryo (2 plots)
REMAINING_EMBRYOS=$((704 - COUNT))
ESTIMATED_NEEDED=$((AVG_PER_EMBRYO * REMAINING_EMBRYOS / 1024))

echo ""
if [ $ESTIMATED_NEEDED -gt $AVAILABLE ]; then
else
fi






