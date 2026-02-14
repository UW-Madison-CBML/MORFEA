"""
检查 frame 数量不匹配的原因
"""
import numpy as np
import pandas as pd
from pathlib import Path


# Load data
tphate_file = "tphate_3d_results.npz"
latents_file = "latents_all_frames.npz"
index_csv = "index.csv"

# Load TPHATE
tphate_data = np.load(tphate_file, allow_pickle=True)
cell_id_tphate = tphate_data['cell_id']
frame_in_cell_tphate = tphate_data['frame_in_cell']

print(f"\nTPHATE Results:")
print(f"  Total points: {len(cell_id_tphate)}")
print(f"  Frame range: {frame_in_cell_tphate.min()} - {frame_in_cell_tphate.max()}")
print(f"  Unique frame_in_cell values: {len(np.unique(frame_in_cell_tphate))}")

# Load latents
latents_data = np.load(latents_file, allow_pickle=True)
cell_id_latents = latents_data['cell_id']
frame_in_cell_latents = latents_data['frame_in_cell']
sequence_idx_latents = latents_data.get('sequence_idx', None)

print(f"\nLatents File:")
print(f"  Total points: {len(cell_id_latents)}")
print(f"  Frame range: {frame_in_cell_latents.min()} - {frame_in_cell_latents.max()}")
print(f"  Unique frame_in_cell values: {len(np.unique(frame_in_cell_latents))}")

# Load index.csv
if Path(index_csv).exists():
    df = pd.read_csv(index_csv)
    print(f"\nIndex.csv:")
    print(f"  Total sequences: {len(df)}")
    
    total_frames_in_sequences = 0
    for idx, row in df.iterrows():
        paths = row['paths'].split('|')
        total_frames_in_sequences += len(paths)
    
    print(f"  Total frames across all sequences: {total_frames_in_sequences}")
    print(f"  Average frames per sequence: {total_frames_in_sequences / len(df):.1f}")
    
    unique_cells = df['cell_id'].unique()
    for cell_id in unique_cells:
        cell_df = df[df['cell_id'] == cell_id]
        print(f"\n  Cell '{cell_id}':")
        print(f"    Sequences: {len(cell_df)}")
        
        cell_frames = set()
        for idx, row in cell_df.iterrows():
            paths = row['paths'].split('|')
            start_idx = row['start_idx']
            for t, path in enumerate(paths):
                frame_idx = start_idx + t
                cell_frames.add(frame_idx)
        
        print(f"    Unique frame indices: {len(cell_frames)}")
        print(f"    Frame range: {min(cell_frames)} - {max(cell_frames)}")
        
        if len(cell_df) > 0:
            first_path = cell_df.iloc[0]['paths'].split('|')[0]
            if 'embryo_dataset' in first_path:
                cell_folder = Path(first_path).parent
                if cell_folder.exists():
                    image_files = list(cell_folder.glob('*.jpeg')) + list(cell_folder.glob('*.jpg')) + list(cell_folder.glob('*.png'))
                    print(f"    Actual images in folder: {len(image_files)}")
                    print(f"    Cell folder: {cell_folder}")


