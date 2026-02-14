#!/bin/bash

cd ~/ivf_repo

echo "=== Verifying TPHATE Results ==="
python3 << 'PYTHON'
import numpy as np

print("1. Checking tphate_3d_results_direct.npz...")
d = np.load('tphate_3d_results_direct.npz', allow_pickle=True)
f = d['frame_in_cell']
Z_tphate = d['Z_tphate']

print(f"   Total frames: {len(f)}")
print(f"   Frame range: {f.min()} - {f.max()}")
print(f"   TPHATE shape: {Z_tphate.shape}")
print(f"   Expected: 435 frames (0-434)")

if len(f) == 435 and f.max() == 434:
    print("   ✓ Frame count is CORRECT!")
else:
    print("   ✗ Frame count is WRONG!")

print("\n2. Checking latents_all_frames_direct.npz...")
d2 = np.load('latents_all_frames_direct.npz', allow_pickle=True)
f2 = d2['frame_in_cell']
print(f"   Total frames: {len(f2)}")
print(f"   Frame range: {f2.min()} - {f2.max()}")

if len(f) == len(f2):
    print("   ✓ Frame counts match!")
else:
    print(f"   ✗ Frame counts don't match! ({len(f)} vs {len(f2)})")

print("\n3. Checking visualization files...")
import os
files = [
    'tphate_segments_direct/tphate_3d_gradient.png',
    'tphate_segments_direct/tphate_3d_segments.png',
    'tphate_segments_direct/segment_A_frames.png',
    'tphate_segments_direct/segment_B_frames.png',
    'tphate_segments_direct/segment_C_frames.png',
    'tphate_segments_direct/segment_D_frames.png',
    'tphate_segments_direct/tphate_segments_combined.png',
    'tphate_segments_direct/all_segments_all_frames.png'
]

for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024  # KB
        print(f"   ✓ {f} ({size:.1f} KB)")
    else:
        print(f"   ✗ {f} NOT FOUND")

print("\n=== Summary ===")
if len(f) == 435 and f.max() == 434:
    print("✅ Frame count: CORRECT (435 frames, 0-434)")
else:
    print("❌ Frame count: WRONG")
    
if all(os.path.exists(f) for f in files):
    print("✅ All visualization files generated")
else:
    print("❌ Some visualization files missing")
PYTHON

echo ""
echo "=== File sizes ==="
ls -lh tphate_segments_direct/*.png 2>/dev/null | tail -10

