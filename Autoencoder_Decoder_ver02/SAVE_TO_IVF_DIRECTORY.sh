#!/bin/bash

cd /staging/groups/bhaskar_group/rho9

ls -ld /staging/groups/bhaskar_group/ivf/

echo ""
mkdir -p /staging/groups/bhaskar_group/ivf/v1_baseline_tphate/tphate_plots
mkdir -p /staging/groups/bhaskar_group/ivf/v1_baseline_tphate/curvature_plots

if [ $? -eq 0 ]; then
    echo ""
    
    COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
    echo ""
    
    python3 generate_tphate_plots.py \
        --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
        --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
        --output_base /staging/groups/bhaskar_group/ivf/v1_baseline_tphate \
        --knn 5 \
        --start_from $COUNT \
        > /tmp/tphate_run.log 2>&1 &
    
else
fi






