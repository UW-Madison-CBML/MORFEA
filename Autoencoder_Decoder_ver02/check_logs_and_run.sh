#!/bin/bash
# 检查 logs 并运行 TPHATE

cd ~/ivf_repo

echo "=== Checking logs directory ==="
ls -lh logs/ | head -20
du -sh logs/

echo ""
echo "=== You can delete old logs if needed ==="
echo "  rm -f logs/training_log_*.json  # old training logs"
echo "  rm -f logs/*.log  # log files"
echo ""

echo "=== Current disk space ==="
df -h ~ | tail -1

echo ""
echo "=== Running TPHATE Pipeline ==="
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
echo "=== Verifying TPHATE Results ==="
python3 << 'PYTHON'
import numpy as np

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
PYTHON

if [ $? -ne 0 ]; then
    echo "❌ Verification failed!"
    exit 1
fi

echo ""
echo "=== Generating Visualizations ==="
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
    echo "Generated files:"
    ls -lh tphate_segments_direct/*.png tphate_segments_direct/*.json 2>/dev/null | tail -10
else
    echo "❌ Visualization failed!"
    exit 1
fi

