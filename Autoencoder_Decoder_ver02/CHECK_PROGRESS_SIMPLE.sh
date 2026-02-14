#!/bin/bash

OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
PROGRESS=$((COUNT - 182))
TIME=$(date '+%H:%M:%S')

echo ""

if [ $COUNT -ge 382 ]; then
elif [ $COUNT -gt 182 ]; then
    REMAINING=$((382 - COUNT))
else
fi
