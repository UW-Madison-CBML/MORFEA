#!/bin/bash

cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

mkdir -p tphate_results_final

echo "=== Downloading TPHATE Results ==="
echo ""

echo "1. Downloading TPHATE results..."
scp rho9@ap2001.chtc.wisc.edu:~/ivf_repo/tphate_3d_results_direct.npz tphate_results_final/ 2>/dev/null && echo "   ✓ tphate_3d_results_direct.npz" || echo "   ✗ tphate_3d_results_direct.npz failed"

echo ""
echo "2. Downloading preprocessed latents..."
scp rho9@ap2001.chtc.wisc.edu:~/ivf_repo/latents_preprocessed_direct.npz tphate_results_final/ 2>/dev/null && echo "   ✓ latents_preprocessed_direct.npz" || echo "   ✗ latents_preprocessed_direct.npz failed"

echo ""
echo "3. Downloading visualization files..."
scp rho9@ap2001.chtc.wisc.edu:~/ivf_repo/tphate_segments_direct/*.png tphate_results_final/ 2>/dev/null && echo "   ✓ All PNG files" || echo "   ✗ PNG files failed"

scp rho9@ap2001.chtc.wisc.edu:~/ivf_repo/tphate_segments_direct/*.json tphate_results_final/ 2>/dev/null && echo "   ✓ JSON metadata" || echo "   ✗ JSON failed"

echo ""
echo "=== Download Complete ==="
echo ""
echo "Files in tphate_results_final/:"
ls -lh tphate_results_final/ | tail -15

echo ""
echo "=== Verifying Files ==="
python3 << 'PYTHON'
import os
import numpy as np

files_to_check = [
    'tphate_results_final/tphate_3d_results_direct.npz',
    'tphate_results_final/latents_all_frames_direct.npz',
    'tphate_results_final/tphate_3d_gradient.png',
    'tphate_results_final/tphate_3d_segments.png',
    'tphate_results_final/segment_A_frames.png',
    'tphate_results_final/segment_B_frames.png',
    'tphate_results_final/segment_C_frames.png',
    'tphate_results_final/segment_D_frames.png',
    'tphate_results_final/tphate_segments_combined.png',
    'tphate_results_final/all_segments_all_frames.png'
]

print("Checking downloaded files...")
for f in files_to_check:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024  # KB
        print(f"  ✓ {os.path.basename(f)} ({size:.1f} KB)")
    else:
        print(f"  ✗ {os.path.basename(f)} NOT FOUND")

if os.path.exists('tphate_results_final/tphate_3d_results_direct.npz'):
    print("\nVerifying frame count...")
    d = np.load('tphate_results_final/tphate_3d_results_direct.npz', allow_pickle=True)
    f = d['frame_in_cell']
    print(f"  Total frames: {len(f)}")
    print(f"  Frame range: {f.min()} - {f.max()}")
    if len(f) == 435 and f.max() == 434:
        print("  ✓ Frame count is CORRECT (435 frames, 0-434)!")
    else:
        print("  ✗ Frame count is WRONG!")
PYTHON

