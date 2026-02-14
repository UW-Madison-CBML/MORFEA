#!/bin/bash

echo ""

condor_q -submitter rho9 | grep -E "(ID|generate_tphate|RUN|IDLE|HOLD)"
echo ""

OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
if [ -d "$OUTPUT_DIR/tphate_plots" ]; then
    TPHATE_COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    CURVATURE_COUNT=$(ls -1 "$OUTPUT_DIR/curvature_plots"/*.png 2>/dev/null | wc -l)
    echo "   T-PHATE plots: $TPHATE_COUNT / 704"
    echo "   Curvature plots: $CURVATURE_COUNT / 704"
    echo ""
    
    if [ $TPHATE_COUNT -gt 0 ]; then
        ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -3 | sed 's/.*\///; s/_tphate\.png//'
    fi
else
fi

echo ""

LATEST_LOG=$(ls -t ~/logs/generate_tphate_*.out 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo ""
    tail -10 "$LATEST_LOG"
else
fi

echo ""






