#!/bin/bash
# 在 CHTC 上运行的完整命令序列

echo "=== Step 1: Force downgrade NumPy to 1.24.3 ==="
# 强制安装 NumPy 1.24.3（稳定版本，兼容 s_gd2）
pip install --user "numpy==1.24.3" --force-reinstall --no-deps --no-cache-dir

echo ""
echo "=== Step 2: Verify NumPy ==="
python3 -c "import numpy as np; print(f'NumPy version: {np.__version__}')"

echo ""
echo "=== Step 3: Reinstall s_gd2 and tphate ==="
pip uninstall --user -y s_gd2 tphate 2>/dev/null || true
pip install --user s_gd2 --no-cache-dir --force-reinstall
pip install --user tphate --no-cache-dir --force-reinstall

echo ""
echo "=== Step 4: Verify tphate ==="
python3 -c "import tphate; print('✓ tphate imported successfully')" || echo "❌ tphate import failed"

echo ""
echo "=== Step 5: Remove old TPHATE files ==="
rm -f latents_preprocessed_direct.npz
rm -f tphate_3d_results_direct.npz

echo ""
echo "=== Step 6: Preprocess (435 frames) ==="
python3 preprocess_latents.py \
    --input latents_all_frames_direct.npz \
    --output latents_preprocessed_direct.npz \
    --pca_components 32

echo ""
echo "=== Step 7: Verify preprocessed ==="
python3 -c "import numpy as np; d=np.load('latents_preprocessed_direct.npz'); f=d['frame_in_cell']; print(f'Preprocessed: {len(f)} frames, range: {f.min()}-{f.max()}')"

echo ""
echo "=== Step 8: Run TPHATE ==="
python3 tphate_3d_pipeline.py \
    --input latents_preprocessed_direct.npz \
    --output tphate_3d_results_direct.npz \
    --use_pca \
    --knn 10 \
    --n_components 3

echo ""
echo "=== Step 9: Verify TPHATE ==="
python3 -c "import numpy as np; d=np.load('tphate_3d_results_direct.npz'); f=d['frame_in_cell']; print(f'TPHATE: {len(f)} frames, range: {f.min()}-{f.max}'); print('Expected: 435 (0-434)'); print('✓ CORRECT!' if len(f) == 435 and f.max() == 434 else '✗ WRONG!')"

echo ""
echo "=== Step 10: Visualize ==="
python3 visualize_tphate_segments.py \
    --tphate_file tphate_3d_results_direct.npz \
    --latents_file latents_all_frames_direct.npz \
    --output_dir tphate_segments_direct \
    --n_segments 4

echo ""
echo "=========================================="
echo "✅ Pipeline Complete!"
echo "=========================================="

