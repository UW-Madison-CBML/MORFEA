#!/bin/bash
# Fix index.csv paths and regenerate reconstructions on CHTC

echo ""

cd ~/ivf_repo

if [ -L "data" ]; then
    ls -ld data
else
    if [ -d "/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset" ]; then
        ln -sf /staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset data
    elif [ -d "/staging/groups/bhaskar_group/ivf/embryo_dataset" ]; then
        ln -sf /staging/groups/bhaskar_group/ivf/embryo_dataset data
    else
        exit 1
    fi
fi

if [ -f "index.csv" ]; then
    echo ""
    cp index.csv index.csv.backup_$(date +%Y%m%d_%H%M%S)
fi

echo ""
python3 build_index.py --root data --out index.csv

if [ ! -f "index.csv" ]; then
    exit 1
fi


echo ""
rm -rf reconstructions/*.png 2>/dev/null

python3 generate_reconstructions.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --index_csv index.csv \
    --output_dir reconstructions \
    --num_samples 5 \
    --n_frames 10

echo ""

