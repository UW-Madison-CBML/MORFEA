#!/bin/bash

ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep
if [ $? -ne 0 ]; then
else
fi
echo ""

if [ -f /staging/groups/bhaskar_group/rho9/tphate_run.log ]; then
    tail -30 /staging/groups/bhaskar_group/rho9/tphate_run.log
else
fi
echo ""

OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
if [ -d "$OUTPUT_DIR/tphate_plots" ]; then
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    
    echo ""
    ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -3 | xargs -n1 basename | sed 's/_tphate\.png//'
else
fi

echo ""
echo ""






