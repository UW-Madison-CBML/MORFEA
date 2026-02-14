#!/bin/bash

cd /staging/groups/bhaskar_group/rho9

COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
echo ""

if ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep > /dev/null; then
    ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep
else
    echo ""
    echo ""
    
    python3 generate_tphate_for_aadhitya.py \
        --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
        --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
        --output_base /staging/groups/bhaskar_group/rho9/v1_baseline_tphate \
        --knn 5 \
        --start_from $COUNT \
        > /tmp/tphate_run.log 2>&1 &
    
fi






