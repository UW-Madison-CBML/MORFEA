#!/bin/bash
# 重新运行 TPHATE pipeline（使用正确的 435-frame latents）

echo "=== Re-running TPHATE Pipeline with 435 frames ==="
echo ""

cd ~/ivf_repo

# 删除旧的预处理和 TPHATE 文件
echo "1. Removing old preprocessed and TPHATE files..."
rm -f latents_preprocessed_direct.npz
rm -f tphate_3d_results_direct.npz
echo "✓ Old files removed"
echo ""

# Step 2: 重新预处理（使用新的 435-frame latents）
echo "2. Preprocessing latents (435 frames)..."
python3 preprocess_latents.py \
    --input latents_all_frames_direct.npz \
    --output latents_preprocessed_direct.npz \
    --pca_components 32

if [ $? -ne 0 ]; then
    echo "❌ Preprocessing failed!"
    exit 1
fi

# 验证预处理后的 frame 数量
echo ""
echo "Verifying preprocessed frame count..."
python3 -c "import numpy as np; d=np.load('latents_preprocessed_direct.npz'); f=d['frame_in_cell']; print(f'Preprocessed frames: {len(f)}, range: {f.min()}-{f.max()}')"
echo ""

# Step 3-4: 重新运行 TPHATE
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

# 验证 TPHATE 结果的 frame 数量
echo ""
echo "Verifying TPHATE frame count..."
python3 -c "import numpy as np; d=np.load('tphate_3d_results_direct.npz'); f=d['frame_in_cell']; print(f'TPHATE frames: {len(f)}, range: {f.min()}-{f.max}'); print('Expected: 435 (0-434)'); print('✓ CORRECT!' if len(f) == 435 and f.max() == 434 else '✗ WRONG!')"
echo ""

# Step 5: 重新可视化
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

