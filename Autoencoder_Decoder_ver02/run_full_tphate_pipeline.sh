#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "=========================================="
echo ""

cd ~/ivf_repo

python3 -c "import torch" 2>/dev/null || {
    echo "⚠️  torch not found, but should be available"
}

python3 -c "import sklearn" 2>/dev/null || {
    echo "⚠️  sklearn not found, installing..."
    pip install --user scikit-learn
}

python3 -c "import phate" 2>/dev/null || {
    echo "⚠️  phate not found, installing..."
    pip install --user phate
}

python3 -c "import tphate" 2>/dev/null && {
    echo "✓ tphate library available (REQUIRED)"
} || {
    echo "❌ ERROR: tphate library is REQUIRED but not available!"
    echo "   Installing tphate..."
    pip install --user tphate
    if [ $? -ne 0 ]; then
        echo "   Installation failed. Please install manually: pip install tphate"
        exit 1
    fi
    echo "✓ tphate installed"
}

echo ""

# Step 1: Export all frame latents (direct from cell folders)
echo "=========================================="
echo "Step 1: Export All Frame Latents (Direct)"
echo "=========================================="
echo "  Using export_all_frame_latents_direct.py to extract ALL frames"
echo "  (bypassing index.csv subsampling)"
echo ""
python3 export_all_frame_latents_direct.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --data_root data \
    --output latents_all_frames_direct.npz \
    --cell_ids RI382-2

if [ $? -ne 0 ]; then
    echo "❌ Step 1 failed!"
    exit 1
fi

echo ""
echo "✓ Step 1 complete"
echo ""

# Step 2: Preprocess latents
echo "=========================================="
echo "Step 2: Preprocess Latents"
echo "=========================================="
python3 preprocess_latents.py \
    --input latents_all_frames_direct.npz \
    --output latents_preprocessed_direct.npz \
    --pca_components 32 \
    --outlier_threshold 5.0

if [ $? -ne 0 ]; then
    echo "❌ Step 2 failed!"
    exit 1
fi

echo ""
echo "✓ Step 2 complete"
echo ""

# Step 3-4: 3D TPHATE (REQUIRED, no approximation)
echo "=========================================="
echo "Step 3-4: 3D TPHATE (REQUIRED)"
echo "=========================================="
python3 tphate_3d_pipeline.py \
    --input latents_preprocessed_direct.npz \
    --output tphate_3d_results_direct.npz \
    --use_pca \
    --knn 10 \
    --n_components 3 \
    --seed 42

if [ $? -ne 0 ]; then
    echo "❌ Step 3-4 failed!"
    exit 1
fi

echo ""
echo "✓ Step 3-4 complete"
echo ""

echo "=========================================="
echo "✅ Pipeline Complete!"
echo "=========================================="
echo ""
echo ""
echo "  - TDA (persistent homology)"
echo "  - Feature extraction for downstream classifier"
echo ""

