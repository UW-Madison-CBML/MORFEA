#!/bin/bash

echo ""

cd ~/ivf_repo

if [ ! -f "model.py" ]; then
    exit 1
fi

if [ ! -f "checkpoints/checkpoint_epoch_50.pt" ]; then
    exit 1
fi

if [ ! -L "data" ] && [ ! -d "data" ]; then
    if [ -d "/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset" ]; then
        ln -sf /staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset data
    elif [ -d "/staging/groups/bhaskar_group/ivf/embryo_dataset" ]; then
        ln -sf /staging/groups/bhaskar_group/ivf/embryo_dataset data
    else
        exit 1
    fi
fi

if [ ! -f "index.csv" ] || [ $(head -1 index.csv | grep -c "/var/lib/condor") -gt 0 ]; then
    python3 build_index.py --root data --out index.csv
fi

echo ""
python3 generate_reconstructions.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --index_csv index.csv \
    --output_dir reconstructions \
    --num_samples 5 \
    --n_frames 10

echo ""

