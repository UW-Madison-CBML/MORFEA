#!/bin/bash

echo "=== Re-running TPHATE Pipeline with 435 frames ==="
echo ""

cd ~/ivf_repo

echo "1. Removing old preprocessed and TPHATE files..."
rm -f latents_preprocessed_direct.npz
rm -f tphate_3d_results_direct.npz
echo "✓ Old files removed"
echo ""

echo "2. Preprocessing latents (435 frames)..."
python3 preprocess_latents.py \
    --input latents_all_frames_direct.npz \
    --output latents_preprocessed_direct.npz \
    --pca_components 32

if [ $? -ne 0 ]; then
    echo "❌ Preprocessing failed!"
    exit 1
fi

echo ""
echo "Verifying preprocessed frame count..."
python3 -c "import numpy as np; d=np.load('latents_preprocessed_direct.npz'); f=d['frame_in_cell']; print(f'Preprocessed frames: {len(f)}, range: {f.min()}-{f.max()}')"
echo ""

echo "3. Running TPHATE (435 frames)..."
python3 tphate_3d_pipeline.py \
    --input latents_preprocessed_direct.npz \
    --output tphate_3d_results_direct.npz \
    --use_pca \
    --knn 10 \
    --n_components 3

if [ $? -ne 0 ]; then
    echo "❌ TPHATE failed!"
    exit 1
fi

echo ""
echo "Verifying TPHATE frame count..."
python3 -c "import numpy as np; d=np.load('tphate_3d_results_direct.npz'); f=d['frame_in_cell']; print(f'TPHATE frames: {len(f)}, range: {f.min()}-{f.max}'); print('Expected: 435 (0-434)'); print('✓ CORRECT!' if len(f) == 435 and f.max() == 434 else '✗ WRONG!')"
echo ""

echo "4. Re-generating visualizations..."
python3 visualize_tphate_segments.py \
    --tphate_file tphate_3d_results_direct.npz \
    --latents_file latents_all_frames_direct.npz \
    --output_dir tphate_segments_direct \
    --n_segments 4

echo ""
echo "=========================================="
echo "✅ Pipeline Complete!"
echo "=========================================="

