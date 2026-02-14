#!/bin/bash

cd /staging/groups/bhaskar_group/rho9

echo ""

python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/ivf/v1_baseline_tphate \
    --knn 5 \
    2>&1 | head -50

echo ""






