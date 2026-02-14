#!/bin/bash

cd /staging/groups/bhaskar_group/rho9

pkill -f "generate_tphate_for_aadhitya.py" 2>/dev/null

nohup python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_val \
    --knn 5 \
    > tphate_run.log 2>&1 &







