#!/bin/bash
# Check rho9/ivf_data directory structure

echo ""

DATA_DIR="/staging/groups/bhaskar_group/rho9/ivf_data"

if [ -d "$DATA_DIR" ]; then
    ls -lh "$DATA_DIR"
    echo ""
    
    if [ -d "$DATA_DIR/embryo_dataset" ]; then
        echo ""
        ls -d "$DATA_DIR/embryo_dataset"/*/ 2>/dev/null | head -5
        echo ""
        find "$DATA_DIR/embryo_dataset" -name "*.jpeg" -type f 2>/dev/null | head -3
    else
        echo ""
        ls -lh "$DATA_DIR"/*.tar.gz 2>/dev/null
    fi
else
fi

