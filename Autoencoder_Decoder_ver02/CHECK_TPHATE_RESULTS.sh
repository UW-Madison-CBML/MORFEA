#!/bin/bash

echo ""

BASE_DIR="aadhitya_v1_test"

if [ -d "$BASE_DIR" ]; then
    echo ""
    
    if [ -d "$BASE_DIR/tphate_plots" ]; then
        echo "T-PHATE plots:"
        ls -lh "$BASE_DIR/tphate_plots/"*.png 2>/dev/null | head -10
        COUNT=$(ls -1 "$BASE_DIR/tphate_plots/"*.png 2>/dev/null | wc -l)
        echo ""
    else
    fi
    
    if [ -d "$BASE_DIR/curvature_plots" ]; then
        echo "Curvature plots:"
        ls -lh "$BASE_DIR/curvature_plots/"*.png 2>/dev/null | head -10
        COUNT=$(ls -1 "$BASE_DIR/curvature_plots/"*.png 2>/dev/null | wc -l)
        echo ""
    else
    fi
    
    du -sh "$BASE_DIR/tphate_plots" 2>/dev/null
    du -sh "$BASE_DIR/curvature_plots" 2>/dev/null
    
else
fi






