"""
检查 TPHATE 结果中的 cell_id 和对应的数据
"""
import numpy as np
import pandas as pd
from pathlib import Path

# Load TPHATE data
tphate_file = "tphate_3d_results.npz"
latents_file = "latents_all_frames.npz"
index_csv = "index.csv"

print("=== Checking TPHATE Cell IDs ===")

# Load TPHATE
tphate_data = np.load(tphate_file, allow_pickle=True)
cell_id_tphate = tphate_data['cell_id']
frame_in_cell_tphate = tphate_data['frame_in_cell']
sequence_idx_tphate = tphate_data.get('sequence_idx', None)

print(f"\nTPHATE Results:")
print(f"  Total points: {len(cell_id_tphate)}")
print(f"  Unique cells: {np.unique(cell_id_tphate)}")
print(f"  Frame range: {frame_in_cell_tphate.min()} - {frame_in_cell_tphate.max()}")
if sequence_idx_tphate is not None:
    print(f"  Sequence idx range: {sequence_idx_tphate.min()} - {sequence_idx_tphate.max()}")

# Load latents
latents_data = np.load(latents_file, allow_pickle=True)
cell_id_latents = latents_data['cell_id']
sequence_idx_latents = latents_data.get('sequence_idx', None)

print(f"\nLatents File:")
print(f"  Total points: {len(cell_id_latents)}")
print(f"  Unique cells: {np.unique(cell_id_latents)}")
if sequence_idx_latents is not None:
    print(f"  Sequence idx range: {sequence_idx_latents.min()} - {sequence_idx_latents.max()}")

# Load index.csv
if Path(index_csv).exists():
    df = pd.read_csv(index_csv)
    print(f"\nIndex.csv:")
    print(f"  Total sequences: {len(df)}")
    print(f"  Unique cells: {df['cell_id'].unique()}")
    print(f"\nFirst 5 sequences:")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        print(f"  [{i}] cell_id={row['cell_id']}, start_idx={row['start_idx']}, n_paths={len(row['paths'].split('|'))}")
    
    tphate_cell = np.unique(cell_id_tphate)[0] if len(np.unique(cell_id_tphate)) > 0 else None
    if tphate_cell:
        matching = df[df['cell_id'] == tphate_cell]
        if len(matching) > 0:
            first_paths = matching.iloc[0]['paths'].split('|')
            print(f"    {first_paths[0] if len(first_paths) > 0 else 'N/A'}")
else:
    print(f"\n⚠️  index.csv not found!")

