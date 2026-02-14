#!/bin/bash

OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
LOG_FILE="/staging/groups/bhaskar_group/rho9/tphate_run.log"

echo ""

LAST_COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
echo ""

while true; do
    CURRENT_COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    CURRENT_TIME=$(date '+%H:%M:%S')
    
    INCREASED=$((CURRENT_COUNT - LAST_COUNT))
    
    if [ $INCREASED -gt 0 ]; then
        echo "[$CURRENT_TIME] ✓ $CURRENT_COUNT / 704 (+$INCREASED)"
    else
    fi
    
    if [ $INCREASED -gt 0 ] && [ $CURRENT_COUNT -gt 0 ]; then
        LAST_EMBRYO=$(ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -1 | xargs -n1 basename | sed 's/_tphate\.png//')
    fi
    
    LAST_COUNT=$CURRENT_COUNT
    
    if [ $CURRENT_COUNT -ge 704 ]; then
        echo ""
        break
    fi
    
    sleep 30
done






