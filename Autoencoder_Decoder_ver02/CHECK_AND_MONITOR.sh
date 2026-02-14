#!/bin/bash

ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep

echo ""
OUTPUT_DIR="/staging/groups/bhaskar_group/ivf/aadhitya_v1_val"
if [ -d "$OUTPUT_DIR/tphate_plots" ]; then
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    
    if [ $COUNT -gt 0 ]; then
        echo ""
        ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -5 | xargs -n1 basename | sed 's/_tphate\.png//'
    fi
else
fi

echo ""
echo ""

LAST_COUNT=0
while true; do
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    TIME=$(date '+%H:%M:%S')
    
    if [ $COUNT -gt $LAST_COUNT ]; then
        INCREASED=$((COUNT - LAST_COUNT))
        echo "[$TIME] ✓ $COUNT plots (+$INCREASED)"
        LAST_COUNT=$COUNT
    else
    fi
    
    if ! ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep > /dev/null; then
        echo ""
        break
    fi
    
    sleep 30
done






