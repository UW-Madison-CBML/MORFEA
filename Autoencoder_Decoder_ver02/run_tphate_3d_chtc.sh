#!/bin/bash

echo ""

if [ ! -d ~/ivf_repo ]; then
    exit 1
fi

cd ~/ivf_repo || exit 1

if python3 -c "import phate" 2>/dev/null; then
    echo "✓ phate library available"
else
    echo "⚠️  phate library not found, installing..."
    pip install --user phate
fi

if python3 -c "import matplotlib" 2>/dev/null; then
    echo "✓ matplotlib available"
else
    echo "⚠️  matplotlib not found, installing..."
    pip install --user matplotlib
fi

if python3 -c "import sklearn" 2>/dev/null; then
    echo "✓ sklearn available"
else
    echo "⚠️  sklearn not found, installing..."
    pip install --user scikit-learn
fi

echo ""
python3 extract_latents_epoch50.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --index_csv index.csv \
    --output_dir latents_epoch50 \
    --batch_size 8 \
    --use_z_seq

echo ""
python3 tphate_3d_visualization.py \
    --latents_file latents_epoch50/latents_z_seq_epoch50.npy \
    --metadata_file latents_epoch50/latents_metadata_epoch50.json \
    --output_dir tphate_3d_results \
    --n_embryos 10 \
    --knn 5 \
    --decay 40

echo ""
echo "  - latents_epoch50/ (latent vectors)"
echo "  - tphate_3d_results/ (3D visualizations)"

