#!/bin/bash
# 检查环境并运行 pipeline

cd ~/ivf_repo

echo "=== Checking Environment ==="
python3 -c "import numpy as np; print(f'NumPy: {np.__version__}')"
python3 -c "import tphate; print('✓ tphate OK')" 2>&1 || echo "❌ tphate failed"

echo ""
echo "=== Running Pipeline ==="

# 删除旧文件
rm -f latents_preprocessed_direct.npz tphate_3d_results_direct.npz

# 预处理
echo "1. Preprocessing..."
python3 preprocess_latents.py \
    --input latents_all_frames_direct.npz \
    --output latents_preprocessed_direct.npz \
    --pca_components 32

# 验证预处理
echo ""
echo "2. Verifying preprocessed..."
python3 -c "import numpy as np; d=np.load('latents_preprocessed_direct.npz'); f=d['frame_in_cell']; print(f'  Frames: {len(f)}, range: {f.min()}-{f.max()}')"

# TPHATE
echo ""
echo "3. Running TPHATE..."
python3 tphate_3d_pipeline.py \
    --input latents_preprocessed_direct.npz \
    --output tphate_3d_results_direct.npz \
    --use_pca \
    --knn 10 \
    --n_components 3

# 验证 TPHATE
echo ""
echo "4. Verifying TPHATE..."
python3 -c "
import numpy as np
d = np.load('tphate_3d_results_direct.npz')
f = d['frame_in_cell']
print(f'  Frames: {len(f)}, range: {f.min()}-{f.max()}')
print(f'  Expected: 435 (0-434)')
if len(f) == 435 and f.max() == 434:
    print('  ✓ CORRECT!')
else:
    print('  ✗ WRONG!')
"

# 可视化
echo ""
echo "5. Generating visualizations..."
python3 visualize_tphate_segments.py \
    --tphate_file tphate_3d_results_direct.npz \
    --latents_file latents_all_frames_direct.npz \
    --output_dir tphate_segments_direct \
    --n_segments 4

echo ""
echo "=========================================="
echo "✅ Done! Check tphate_segments_direct/ for results"
echo "=========================================="

