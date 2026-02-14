"""
Export All Frame Latents - 直接从 cell folder 读取（不依赖 index.csv）
- 直接从 cell folder 读取所有图片
- 不经过 subsample 和滑动窗口
- 每一帧都提取一次
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from collections import defaultdict
import re
from PIL import Image

# Import model
try:
    from model import Encoder, Decoder
    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()
        
        def forward(self, x):
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            z_seq = self.encoder(x_flat)
            z_seq = z_seq.view(B, T, -1) if len(z_seq.shape) == 2 else z_seq.view(B, T, *z_seq.shape[1:])
            recon = self.decoder(z_seq.view(B*T, -1))
            recon = recon.view(B, T, C, H, W)
            return {
                'reconstruction': recon,
                'z_seq': z_seq,
                'z_last': z_seq[:, -1, :] if len(z_seq.shape) == 3 else z_seq[:, -1]
            }
    
    CHTC_MODEL_AVAILABLE = True
    print("Using CHTC model structure (Encoder + Decoder)")
except (ImportError, AttributeError):
    print("Using local model structure (ConvLSTMAutoencoder)")
    CHTC_MODEL_AVAILABLE = False
    from model import ConvLSTMAutoencoder


def extract_frame_latent_from_encoder(model, frame, device="cpu"):
    model.eval()
    with torch.no_grad():
        frame = frame.to(device)
        
        if hasattr(model, 'frame_encoder'):
            z = model.frame_encoder(frame)
            return z.squeeze(0).cpu().numpy()
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'frame_encoder'):
            z = model.encoder.frame_encoder(frame)
            return z.squeeze(0).cpu().numpy()
        else:
            seq = frame.unsqueeze(0)
            output = model(seq)
            z_seq = output['z_seq']
            
            if len(z_seq.shape) == 5:
                B, T, C, H, W = z_seq.shape
                z_seq = z_seq.view(B, T, C, -1).mean(dim=-1)
            elif len(z_seq.shape) == 4:
                B, T, C, H = z_seq.shape
                z_seq = z_seq.view(B, T, C, -1).mean(dim=-1)
            
            z = z_seq.squeeze(0).squeeze(0)
            return z.cpu().detach().numpy()


def parse_time_from_path(path_str):
    run_match = re.search(r'RUN[_\- ]?(\d+)', path_str, re.I)
    if run_match:
        return int(run_match.group(1))
    nums = re.findall(r'\d+', Path(path_str).name)
    if nums:
        return int(nums[-1])
    return 0


def list_all_frames_in_cell(cell_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    frames = []
    for ext in exts:
        frames += list(cell_dir.glob(ext))
    frames = [p for p in frames if p.exists() and p.stat().st_size > 0]
    
    run_pat = re.compile(r'RUN[_\- ]?(\d+)', re.I)
    num_pat = re.compile(r'(\d+)')
    
    def parse_sort_key(p):
        name = p.name
        run_m = run_pat.search(name)
        run_idx = int(run_m.group(1)) if run_m else 10**9
        nums = [int(x) for x in num_pat.findall(name)]
        nums = tuple(nums) if nums else ()
        mtime = p.stat().st_mtime_ns
        return (run_idx, nums, mtime)
    
    frames.sort(key=parse_sort_key)
    return frames


def export_all_frame_latents_direct(
    checkpoint_path="checkpoints/checkpoint_epoch_50.pt",
    data_root="data",
    output_file="latents_all_frames_direct.npz",
    device="cuda" if torch.cuda.is_available() else "cpu",
    cell_ids=None
):
    """
    直接从 cell folders 读取所有 frames（不经过 index.csv）
    """
    print(f"=== Export All Frame Latents (Direct from Cell Folders) ===")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    seq_len = config.get('seq_len', 16)
    encoder_hidden_dim = config.get('encoder_hidden_dim', 256)
    
    # Load model
    print("Loading model...")
    chtc_available = CHTC_MODEL_AVAILABLE
    use_chtc_model = False
    
    if chtc_available:
        try:
            model = Autoencoder()
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("✓ Model loaded (CHTC structure)")
            use_chtc_model = True
        except Exception as e:
            print(f"Error loading CHTC model: {e}")
            use_chtc_model = False
    
    if not use_chtc_model:
        from model import ConvLSTMAutoencoder
        model = ConvLSTMAutoencoder(
            seq_len=seq_len,
            input_channels=1,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_layers=2,
            decoder_hidden_dim=128,
            decoder_layers=2,
            use_classifier=False
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✓ Model loaded (local structure)")
    
    model.to(device)
    model.eval()
    
    # Find cell directories
    print(f"\nScanning data root: {data_root}")
    data_path = Path(data_root)
    
    if not data_path.exists():
        if Path('data').exists():
            data_path = Path('data')
            print(f"  Using 'data' symlink: {data_path}")
        else:
            staging_path = Path('/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset')
            if staging_path.exists():
                data_path = staging_path
                print(f"  Using staging path: {data_path}")
            else:
                raise FileNotFoundError(f"Data root not found: {data_root}")
    
    cell_dirs = [p for p in data_path.iterdir() if p.is_dir()]
    if cell_ids:
        cell_dirs = [p for p in cell_dirs if p.name in cell_ids]
    
    print(f"Found {len(cell_dirs)} cell directories")
    
    # Extract all frame latents
    print("\nExtracting frame latents from all cells...")
    all_latents = []
    all_cell_ids = []
    all_frame_in_cell = []
    all_abs_time = []
    all_paths = []
    
    cell_frame_count = defaultdict(int)
    
    with torch.no_grad():
        for cell_dir in cell_dirs:
            cell_id = cell_dir.name
            print(f"\nProcessing cell: {cell_id}")
            
            frames = list_all_frames_in_cell(cell_dir)
            print(f"  Found {len(frames)} frames")
            
            if len(frames) == 0:
                print(f"  ⚠️  No frames found, skipping")
                continue
            
            max_frames = 435
            original_count = len(frames)
            
            if len(frames) > max_frames:
                frames = frames[:max_frames]
                print(f"  ✓ LIMITED to first {max_frames} frames (indices 0-{max_frames-1})")
                print(f"  ✓ EXCLUDING frames {max_frames} to {original_count-1} ({original_count - max_frames} frames skipped)")
            else:
                print(f"  ✓ Processing all {len(frames)} frames (less than {max_frames})")
            
            empty_well_detected = False
            valid_frame_count = 0
            
            for frame_idx, frame_path in enumerate(tqdm(frames, desc=f"  Extracting {cell_id}", total=len(frames))):
                try:
                    img = Image.open(frame_path)
                    img = img.convert("L")
                    img = img.resize((128, 128), Image.BILINEAR)
                    img_array = np.array(img, dtype=np.float32)
                    
                    img_std = img_array.std()
                    img_mean = img_array.mean()
                    img_range = img_array.max() - img_array.min()
                    
                    is_empty = (img_std < 5.0) or (img_range < 10.0)
                    
                    if is_empty:
                        if not empty_well_detected:
                            print(f"\n    ⚠️  Empty well detected at frame {frame_idx}, skipping remaining frames")
                            empty_well_detected = True
                        continue
                    
                    p1, p99 = np.percentile(img_array, [1, 99])
                    if p99 > p1:
                        img_array = np.clip((img_array - p1) / (p99 - p1 + 1e-6), 0, 1)
                    else:
                        img_array = np.clip(img_array / 255.0, 0, 1)
                    
                    frame_tensor = torch.from_numpy(img_array.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    
                    z = extract_frame_latent_from_encoder(model, frame_tensor, device)
                    
                    abs_time = parse_time_from_path(str(frame_path))
                    
                    all_latents.append(z)
                    all_cell_ids.append(cell_id)
                    all_frame_in_cell.append(valid_frame_count)
                    all_abs_time.append(abs_time)
                    all_paths.append(str(frame_path))
                    
                    valid_frame_count += 1
                    cell_frame_count[cell_id] = valid_frame_count
                    
                    if valid_frame_count >= max_frames:
                        print(f"\n    ✓ Reached EXACTLY {max_frames} valid frames (frame_in_cell: 0-{max_frames-1}), stopping")
                        break
                    
                except Exception as e:
                    print(f"    ⚠️  Error loading frame {frame_idx} ({frame_path.name}): {e}")
                    continue
    
    # Check if we extracted any frames
    if len(all_latents) == 0:
        print("\n❌ ERROR: No frames were extracted!")
        return None, None, None, None
    
    # Convert to numpy arrays
    Z = np.array(all_latents)
    cell_id_array = np.array(all_cell_ids)
    frame_in_cell_array = np.array(all_frame_in_cell)
    abs_time_array = np.array(all_abs_time)
    
    print(f"\n✓ Extracted {len(Z)} frames")
    if len(Z) > 0:
        print(f"  Latent dimension: {Z.shape[1]}")
        print(f"  Unique cells: {len(np.unique(cell_id_array))}")
        print(f"  Frame range: {frame_in_cell_array.min()} - {frame_in_cell_array.max()}")
        print(f"  Expected: EXACTLY 435 frames (0-434)")
        if len(Z) != 435:
            print(f"  ⚠️  WARNING: Expected 435 frames but got {len(Z)} frames!")
        if frame_in_cell_array.max() >= 435:
            print(f"  ⚠️  WARNING: Frame index exceeds 434! Max frame: {frame_in_cell_array.max()}")
        for cid in np.unique(cell_id_array):
            count = (cell_id_array == cid).sum()
            print(f"    {cid}: {count} frames")
    
    # Save to .npz
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        Z=Z,
        cell_id=cell_id_array,
        frame_in_cell=frame_in_cell_array,
        abs_time=abs_time_array,
        paths=all_paths
    )
    
    print(f"\n✓ Saved to: {output_file}")
    
    # Save metadata
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get('epoch', 50)),
        "data_root": str(data_root),
        "total_frames": int(len(Z)),
        "latent_dim": int(Z.shape[1]) if len(Z) > 0 else 0,
        "unique_cells": [str(cid) for cid in np.unique(cell_id_array)],
        "cell_frame_counts": {str(k): int(v) for k, v in cell_frame_count.items()},
        "model_config": config
    }
    
    metadata_file = output_path.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to: {metadata_file}")
    
    return Z, cell_id_array, frame_in_cell_array, abs_time_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export all frame latents directly from cell folders")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch_50.pt",
                       help="Path to checkpoint file")
    parser.add_argument("--data_root", type=str, default="data",
                       help="Root directory containing cell folders")
    parser.add_argument("--output", type=str, default="latents_all_frames_direct.npz",
                       help="Output .npz file path")
    parser.add_argument("--cell_ids", type=str, nargs='+', default=None,
                       help="Specific cell IDs to process (None = all)")
    
    args = parser.parse_args()
    
    export_all_frame_latents_direct(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_file=args.output,
        cell_ids=args.cell_ids
    )

