#!/bin/bash

echo ""

CURRENT_DIR=$(pwd)
echo ""

OUTPUT_DIRS=(
    "aadhitya_v1_test"
    "/staging/groups/bhaskar_group/rho9/aadhitya_v1_test"
    "$HOME/aadhitya_v1_test"
)

for dir in "${OUTPUT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo ""
        
        if [ -d "$dir/tphate_plots" ]; then
        fi
        if [ -d "$dir/curvature_plots" ]; then
        fi
        echo ""
    fi
done






