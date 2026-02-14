#!/bin/bash

OUTPUT_DIR="/staging/groups/bhaskar_group/ivf/aadhitya_v1_val"

if ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep > /dev/null; then
    ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep | awk '{print "  PID: "$2", CPU: "$3"%"}'
else
fi

echo ""
if [ -d "$OUTPUT_DIR/tphate_plots" ]; then
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    
    if [ $COUNT -gt 0 ]; then
        echo ""
        ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -3 | xargs -n1 basename | sed 's/_tphate\.png//'
    else
    fi
else
fi






