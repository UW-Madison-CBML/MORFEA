#!/bin/bash

echo "=========================================="
echo "=========================================="
echo ""

condor_q | head -5
echo ""

RESULT_DIR="/staging/groups/bhaskar_group/rho9/model_latents/v1_baseline"
if [ -d "$RESULT_DIR" ]; then
    EMBRYO_COUNT=$(ls -1 "$RESULT_DIR/latents"/*.npy 2>/dev/null | wc -l)
    
    if [ -f "$RESULT_DIR/metadata.json" ]; then
        SUCCESS=$(grep -o '"successful":[0-9]*' "$RESULT_DIR/metadata.json" | grep -o '[0-9]*' || echo "0")
        FAILED=$(grep -o '"failed":[0-9]*' "$RESULT_DIR/metadata.json" | grep -o '[0-9]*' || echo "0")
    fi
    
    echo ""
    ls -lht "$RESULT_DIR/latents"/*.npy 2>/dev/null | head -5 | awk '{print "    " $9 " (" $5 ")"}'
else
fi
echo ""

if [ -f ~/logs/extract_latents_v1_baseline.out ]; then
    tail -20 ~/logs/extract_latents_v1_baseline.out
else
fi
echo ""

if [ -f ~/logs/extract_latents_v1_baseline.err ]; then
    tail -20 ~/logs/extract_latents_v1_baseline.err
else
fi
echo ""

condor_history -limit 3 | head -20
echo ""

echo "=========================================="

