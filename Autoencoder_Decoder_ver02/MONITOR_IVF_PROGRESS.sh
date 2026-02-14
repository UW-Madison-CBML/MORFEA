#!/bin/bash

ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep

echo ""
OUTPUT_DIR="/staging/groups/bhaskar_group/ivf/aadhitya_v1_val"

if [ ! -d "$OUTPUT_DIR/tphate_plots" ]; then
    sleep 10
fi

echo ""

LAST_COUNT=0
ITERATION=0

while true; do
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l 2>/dev/null || echo "0")
    TIME=$(date '+%H:%M:%S')
    ITERATION=$((ITERATION + 1))
    
    if [ "$COUNT" -gt "$LAST_COUNT" ]; then
        INCREASED=$((COUNT - LAST_COUNT))
        LAST_COUNT=$COUNT
    else
        if [ $((ITERATION % 5)) -eq 0 ]; then
        fi
    fi
    
    if ! ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep > /dev/null; then
        echo ""
        break
    fi
    
    if [ "$COUNT" -ge 200 ]; then
        echo ""
        break
    fi
    
    sleep 30
done

echo ""
ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -5 | xargs -n1 basename | sed 's/_tphate\.png//'






