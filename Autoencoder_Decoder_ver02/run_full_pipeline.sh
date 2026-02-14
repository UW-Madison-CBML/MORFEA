#!/bin/bash

cd ~/ivf_repo

echo "=== Step 1: Running TPHATE ==="
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
echo "=== Step 2: Verifying TPHATE Results ==="
python3 << 'PYTHON'
import numpy as np

# Check TPHATE file
print("Checking tphate_3d_results_direct.npz...")
d = np.load('tphate_3d_results_direct.npz', allow_pickle=True)
f = d['frame_in_cell']
Z_tphate = d['Z_tphate']

print(f"  Total frames: {len(f)}")
print(f"  Frame range: {f.min()} - {f.max()}")
print(f"  TPHATE shape: {Z_tphate.shape}")
print(f"  Expected: 435 frames (0-434)")

if len(f) == 435 and f.max() == 434:
    print("  ✓ Frame count is CORRECT!")
else:
    print("  ✗ Frame count is WRONG!")
    exit(1)

# Check latents file for comparison
print("\nChecking latents_all_frames_direct.npz...")
d2 = np.load('latents_all_frames_direct.npz', allow_pickle=True)
f2 = d2['frame_in_cell']
print(f"  Total frames: {len(f2)}")
print(f"  Frame range: {f2.min()} - {f2.max()}")

if len(f) == len(f2):
    print("  ✓ Frame counts match!")
else:
    print(f"  ✗ Frame counts don't match! ({len(f)} vs {len(f2)})")
PYTHON

if [ $? -ne 0 ]; then
    echo "❌ Verification failed!"
    exit 1
fi

echo ""
echo "=== Step 3: Generating Visualizations ==="
python3 visualize_tphate_segments.py \
    --tphate_file tphate_3d_results_direct.npz \
    --latents_file latents_all_frames_direct.npz \
    --output_dir tphate_segments_direct \
    --n_segments 4

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Pipeline Complete!"
    echo "=========================================="
    echo ""
    echo "Generated files in tphate_segments_direct/:"
    ls -lh tphate_segments_direct/*.png tphate_segments_direct/*.json 2>/dev/null | tail -10
else
    echo "❌ Visualization failed!"
    exit 1
fi

