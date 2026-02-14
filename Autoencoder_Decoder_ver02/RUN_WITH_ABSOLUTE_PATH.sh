#!/bin/bash

cd /staging/groups/bhaskar_group/rho9

pkill -f "generate_tphate_for_aadhitya.py" 2>/dev/null
sleep 2

echo ""

nohup python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/rho9/aadhitya_v1_val \
    --knn 5 \
    --start_from 182 \
    --max_embryos 200 \
    > tphate_batch1.log 2>&1 &

echo ""






