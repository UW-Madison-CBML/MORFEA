#!/bin/bash

echo "=== Checking Frame Count in TPHATE Results ==="
echo ""

# python3 << 'PYTHON'
# import numpy as np
# 
# # Load TPHATE results
# tphate_data = np.load('tphate_3d_results_direct.npz', allow_pickle=True)
# frame_in_cell = tphate_data['frame_in_cell']
# 
# print(f"Total frames in TPHATE: {len(frame_in_cell)}")
# print(f"Frame range: {frame_in_cell.min()} - {frame_in_cell.max()}")
# print(f"Unique frame values: {len(np.unique(frame_in_cell))}")
# print(f"Expected: 435 frames (0-434)")
# 
# if len(frame_in_cell) == 435 and frame_in_cell.max() == 434:
#     print("✓ Frame count is CORRECT!")
# else:
#     print("✗ Frame count is WRONG!")
# PYTHON

echo "Run this on CHTC:"
echo "python3 -c \"import numpy as np; d=np.load('tphate_3d_results_direct.npz'); f=d['frame_in_cell']; print(f'Total: {len(f)}, Range: {f.min()}-{f.max()}, Expected: 435 (0-434)')\""
