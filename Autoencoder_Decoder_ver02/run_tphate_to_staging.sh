#!/bin/bash
# 运行 TPHATE pipeline，输出到 staging 目录（避免配额问题）

cd ~/ivf_repo

# 创建 staging 输出目录
STAGING_OUTPUT="/staging/groups/bhaskar_group/rho9/ivf_results"
mkdir -p "$STAGING_OUTPUT"

echo "=== Running TPHATE Pipeline (Output to Staging) ==="
echo "Output directory: $STAGING_OUTPUT"
echo ""

# 运行 TPHATE
python3 tphate_3d_pipeline.py \
    --input latents_preprocessed_direct.npz \
    --output "$STAGING_OUTPUT/tphate_3d_results_direct.npz" \
    --use_pca \
    --knn 10 \
    --n_components 3

if [ $? -ne 0 ]; then
    echo "❌ TPHATE failed!"
    exit 1
fi

echo ""
echo "=== Verifying TPHATE Results ==="
python3 << PYTHON
import numpy as np
import sys

tphate_file = "$STAGING_OUTPUT/tphate_3d_results_direct.npz"
print(f"Checking {tphate_file}...")
d = np.load(tphate_file, allow_pickle=True)
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
    sys.exit(1)
PYTHON

if [ $? -ne 0 ]; then
    echo "❌ Verification failed!"
    exit 1
fi

echo ""
echo "=== Generating Visualizations (to staging) ==="
python3 visualize_tphate_segments.py \
    --tphate_file "$STAGING_OUTPUT/tphate_3d_results_direct.npz" \
    --latents_file latents_all_frames_direct.npz \
    --output_dir "$STAGING_OUTPUT/tphate_segments_direct" \
    --n_segments 4

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Pipeline Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to: $STAGING_OUTPUT"
    echo ""
    ls -lh "$STAGING_OUTPUT"/tphate_*.npz "$STAGING_OUTPUT"/tphate_segments_direct/*.png 2>/dev/null | tail -10
else
    echo "❌ Visualization failed!"
    exit 1
fi

