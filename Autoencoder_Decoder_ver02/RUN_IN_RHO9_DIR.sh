#!/bin/bash

cd /staging/groups/bhaskar_group/rho9

echo ""

mkdir -p /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots
mkdir -p /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/curvature_plots

echo ""
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/rho9/v1_baseline_tphate \
    --knn 5 \
    > /tmp/tphate_run.log 2>&1 &

echo ""
echo "  while true; do"
echo "    COUNT=\$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l || echo 0)"
echo "    echo \"[\$(date '+%H:%M:%S')] \$COUNT / 704 plots\""
echo "    sleep 30"
echo "  done"






