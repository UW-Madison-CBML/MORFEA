#!/bin/bash

ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep
echo ""

if [ -f /staging/groups/bhaskar_group/rho9/tphate_run.log ]; then
    tail -20 /staging/groups/bhaskar_group/rho9/tphate_run.log
else
fi
echo ""

OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
if [ -d "$OUTPUT_DIR/tphate_plots" ]; then
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    echo ""
    
    if [ $COUNT -gt 0 ]; then
        ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -5 | xargs -n1 basename | sed 's/_tphate\.png//'
    fi
else
fi

echo ""






