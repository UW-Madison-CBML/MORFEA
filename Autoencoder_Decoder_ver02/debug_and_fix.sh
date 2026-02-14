#!/bin/bash
# 调试并修复 TPHATE pipeline

cd ~/ivf_repo

echo "=== Step 1: Check what files exist ==="
ls -lh latents_*.npz tphate_*.npz 2>/dev/null || echo "No files found"

echo ""
echo "=== Step 2: Check NumPy and tphate ==="
python3 -c "import numpy as np; print(f'NumPy: {np.__version__}')"
python3 -c "import tphate; print('✓ tphate OK')" 2>&1 || {
    echo "❌ tphate failed, fixing..."
    pip install --user "numpy==1.24.3" --force-reinstall --no-deps --no-cache-dir
    pip uninstall --user -y s_gd2 tphate 2>/dev/null || true
    pip install --user s_gd2 --no-cache-dir --force-reinstall
    pip install --user tphate --no-cache-dir --force-reinstall
    python3 -c "import tphate; print('✓ tphate fixed!')" || echo "❌ Still failed"
}

echo ""
echo "=== Step 3: Check if preprocessed file exists ==="
if [ -f "latents_preprocessed_direct.npz" ]; then
    echo "✓ Preprocessed file exists"
    python3 -c "import numpy as np; d=np.load('latents_preprocessed_direct.npz'); f=d['frame_in_cell']; print(f'  Frames: {len(f)}, range: {f.min()}-{f.max()}')"
else
    echo "❌ Preprocessed file missing, running preprocessing..."
    python3 preprocess_latents.py \
        --input latents_all_frames_direct.npz \
        --output latents_preprocessed_direct.npz \
        --pca_components 32
fi

echo ""
echo "=== Step 4: Run TPHATE ==="
if [ -f "tphate_3d_results_direct.npz" ]; then
    echo "✓ TPHATE file exists, verifying..."
    python3 -c "import numpy as np; d=np.load('tphate_3d_results_direct.npz'); f=d['frame_in_cell']; print(f'  Frames: {len(f)}, range: {f.min()}-{f.max()}')"
else
    echo "❌ TPHATE file missing, running TPHATE..."
    python3 tphate_3d_pipeline.py \
        --input latents_preprocessed_direct.npz \
        --output tphate_3d_results_direct.npz \
        --use_pca \
        --knn 10 \
        --n_components 3
    
    if [ -f "tphate_3d_results_direct.npz" ]; then
        echo "✓ TPHATE completed!"
        python3 -c "import numpy as np; d=np.load('tphate_3d_results_direct.npz'); f=d['frame_in_cell']; print(f'  Frames: {len(f)}, range: {f.min()}-{f.max()}')"
    else
        echo "❌ TPHATE failed! Check error messages above."
        exit 1
    fi
fi

echo ""
echo "=== Step 5: Run visualization ==="
python3 visualize_tphate_segments.py \
    --tphate_file tphate_3d_results_direct.npz \
    --latents_file latents_all_frames_direct.npz \
    --output_dir tphate_segments_direct \
    --n_segments 4

echo ""
echo "=========================================="
echo "✅ Done!"
echo "=========================================="

