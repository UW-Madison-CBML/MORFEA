"""
Step 1: Export All Frame Latents (嚴謹版)
- 用 epoch 50 的 encoder 提取每個 frame 的 latent
- 每個 cell 的每個 frame 都是一個點
- 輸出: Z [N_frames, D], metadata (cell_id, frame_in_cell, abs_time, label)
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

from dataset_ivf import IVFSequenceDataset


def extract_frame_latent_from_encoder(model, frame, device="cpu"):
    """
    從單張 frame 提取 latent vector
    
    Args:
        model: 訓練好的模型
        frame: [1, 1, H, W] 單張 frame
        device: device
    
    Returns:
        z: [D] latent vector
    """
    model.eval()
    with torch.no_grad():
        frame = frame.to(device)
        
        # 方法 1: 如果模型有 frame_encoder，直接用（最直接）
        if hasattr(model, 'frame_encoder'):
            z = model.frame_encoder(frame)  # [1, D]
            return z.squeeze(0).cpu().numpy()  # [D]
        
        # 方法 2: 如果是 ConvLSTMAutoencoder，frame_encoder 在 model 下
        elif hasattr(model, 'frame_encoder'):
            z = model.frame_encoder(frame)  # [1, D]
            return z.squeeze(0).cpu().numpy()  # [D]
        
        # 方法 3: 創建一個單幀的 sequence，通過完整模型
        else:
            # 創建一個單幀的 sequence [1, 1, 1, H, W]
            seq = frame.unsqueeze(0)  # [1, 1, 1, H, W]
            output = model(seq)
            z_seq = output['z_seq']  # [1, 1, ...]
            
            # 處理空間 latent
            if len(z_seq.shape) == 5:
                # [1, 1, C, H, W] -> Global Average Pooling -> [1, 1, C]
                B, T, C, H, W = z_seq.shape
                z_seq = z_seq.view(B, T, C, -1).mean(dim=-1)  # [1, 1, C]
            elif len(z_seq.shape) == 4:
                # [1, 1, C, H] -> Global Average Pooling -> [1, 1, C]
                B, T, C, H = z_seq.shape
                z_seq = z_seq.view(B, T, C, -1).mean(dim=-1)  # [1, 1, C]
            
            z = z_seq.squeeze(0).squeeze(0)  # [D]
            return z.cpu().detach().numpy()


def parse_time_from_path(path_str):
    """
    從路徑中解析時間信息（例如 RUN number）
    
    Args:
        path_str: 圖片路徑
    
    Returns:
        time_value: 時間值（RUN number 或 frame index）
    """
    # 嘗試提取 RUN number
    run_match = re.search(r'RUN[_\- ]?(\d+)', path_str, re.I)
    if run_match:
        return int(run_match.group(1))
    
    # 如果沒有 RUN，嘗試提取其他數字
    nums = re.findall(r'\d+', Path(path_str).name)
    if nums:
        return int(nums[-1])  # 取最後一個數字
    
    return 0


def export_all_frame_latents(
    checkpoint_path="checkpoints/checkpoint_epoch_50.pt",
    index_csv="index.csv",
    output_file="latents_all_frames.npz",
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=32,  # 批次處理 frames
    max_sequences=None  # 限制處理的序列數（None = 全部）
):
    """
    提取所有 frame 的 latent vectors
    
    Args:
        checkpoint_path: checkpoint 路徑
        index_csv: index CSV 路徑
        output_file: 輸出 .npz 文件路徑
        device: device
        batch_size: 批次大小（用於批次處理 frames）
        max_sequences: 最大序列數（None = 全部）
    """
    print(f"=== Step 1: Export All Frame Latents ===")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
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
            print("Falling back to local model structure...")
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
    
    # Load index.csv to get cell information
    print(f"\nLoading index from: {index_csv}")
    df = pd.read_csv(index_csv)
    
    print(f"Total sequences in index.csv: {len(df)}")
    
    # 直接从 cell folders 读取所有 frames（去重，每帧只提取一次）
    print("\nExtracting frame latents (each frame only once, no duplicates)...")
    all_latents = []
    all_cell_ids = []
    all_frame_in_cell = []
    all_abs_time = []
    all_sequence_idx = []
    all_paths = []
    
    # 統計每個 cell 的 frame 數
    cell_frame_count = defaultdict(int)
    
    # 按 cell_id 分组，收集所有唯一的 frames
    unique_cells = df['cell_id'].unique()
    print(f"Unique cells: {unique_cells}")
    
    with torch.no_grad():
        for cell_id in unique_cells:
            print(f"\nProcessing cell: {cell_id}")
            cell_df = df[df['cell_id'] == cell_id]
            
            # 收集这个 cell 的所有 frames（去重）
            # 使用 dict 确保每个 frame_in_cell 只出现一次
            cell_frames_dict = {}  # {frame_in_cell: (path, abs_time, seq_idx)}
            
            for seq_idx, row in cell_df.iterrows():
                paths = row['paths'].split('|')
                start_idx = row['start_idx']
                
                for t, path in enumerate(paths):
                    frame_in_cell = start_idx + t
                    # 只保留第一次出现的 frame（去重）
                    if frame_in_cell not in cell_frames_dict:
                        abs_time = parse_time_from_path(path)
                        cell_frames_dict[frame_in_cell] = (path, abs_time, seq_idx)
            
            # 按 frame_in_cell 排序
            sorted_frames = sorted(cell_frames_dict.items())
            print(f"  Total unique frames: {len(sorted_frames)}")
            if len(sorted_frames) > 0:
                print(f"  Frame range: {sorted_frames[0][0]} - {sorted_frames[-1][0]}")
            
            # 提取每个 frame 的 latent（只提取一次）
            for frame_in_cell, (path, abs_time, seq_idx) in tqdm(sorted_frames, desc=f"  Extracting {cell_id}"):
                # 加载图片
                try:
                    # 尝试修复路径
                    img_path = str(path)
                    
                    # 路径修复策略（按优先级）
                    paths_to_try = [img_path]
                    
                    # 策略1: 去掉 /mnt/htc-cephfs/fuse/root/ 前缀
                    if '/mnt/htc-cephfs/fuse/root' in img_path:
                        paths_to_try.append(img_path.replace('/mnt/htc-cephfs/fuse/root', ''))
                    
                    # 策略2: 使用 data 符号链接
                    if 'embryo_dataset/' in img_path:
                        rel_path = img_path.split('embryo_dataset/')[-1]
                        paths_to_try.append(str(Path('data') / rel_path))
                        # 也尝试去掉前缀后的 data 链接
                        if '/mnt/htc-cephfs/fuse/root' in img_path:
                            paths_to_try.append(str(Path('data') / rel_path))
                    
                    # 尝试所有路径
                    img_path_found = None
                    for p in paths_to_try:
                        if Path(p).exists():
                            img_path_found = p
                            break
                    
                    if img_path_found is None:
                        raise FileNotFoundError(f"Image not found in any of the tried paths: {paths_to_try[:2]}")
                    
                    img_path = img_path_found
                    
                    # 加载并预处理图片
                    img = Image.open(img_path)
                    img = img.convert("L")
                    img = img.resize((128, 128), Image.BILINEAR)
                    img_array = np.array(img, dtype=np.float32)
                    
                    # 归一化（minmax01）
                    p1, p99 = np.percentile(img_array, [1, 99])
                    if p99 > p1:
                        img_array = np.clip((img_array - p1) / (p99 - p1 + 1e-6), 0, 1)
                    else:
                        img_array = np.clip(img_array / 255.0, 0, 1)
                    
                    # 转换为 tensor [1, 1, H, W]，确保是 float32
                    frame_tensor = torch.from_numpy(img_array.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    
                    # 提取 latent
                    z = extract_frame_latent_from_encoder(model, frame_tensor, device)
                    
                    all_latents.append(z)
                    all_cell_ids.append(cell_id)
                    all_frame_in_cell.append(frame_in_cell)
                    all_abs_time.append(abs_time)
                    all_sequence_idx.append(seq_idx)
                    all_paths.append(path)
                    
                    cell_frame_count[cell_id] = max(cell_frame_count[cell_id], frame_in_cell + 1)
                    
                except Exception as e:
                    print(f"    ⚠️  Error loading frame {frame_in_cell} ({path[:60]}...): {e}")
                    # 跳过这个 frame
                    continue
    
    # Check if we extracted any frames
    if len(all_latents) == 0:
        print("\n❌ ERROR: No frames were extracted!")
        print("   All frames failed to load. Check image paths and model compatibility.")
        return None, None, None, None
    
    # Convert to numpy arrays
    Z = np.array(all_latents)  # [N_frames, D]
    cell_id_array = np.array(all_cell_ids)
    frame_in_cell_array = np.array(all_frame_in_cell)
    abs_time_array = np.array(all_abs_time)
    sequence_idx_array = np.array(all_sequence_idx)
    
    print(f"\n✓ Extracted {len(Z)} frames")
    if len(Z) > 0:
        print(f"  Latent dimension: {Z.shape[1]}")
        print(f"  Unique cells: {len(np.unique(cell_id_array))}")
        print(f"  Average frames per cell: {len(Z) / len(np.unique(cell_id_array)):.1f}")
    else:
        print("  ⚠️  No frames extracted!")
    
    # Save to .npz
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        Z=Z,
        cell_id=cell_id_array,
        frame_in_cell=frame_in_cell_array,
        abs_time=abs_time_array,
        sequence_idx=sequence_idx_array,
        paths=all_paths
    )
    
    print(f"\n✓ Saved to: {output_file}")
    
    # Save metadata JSON
    # Convert numpy types to Python native types for JSON serialization
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get('epoch', 50)),
        "index_csv": str(index_csv),
        "total_frames": int(len(Z)),
        "latent_dim": int(Z.shape[1]),
        "unique_cells": int(len(np.unique(cell_id_array))),
        "model_config": {k: (int(v) if isinstance(v, (np.integer, np.int64, np.int32)) else 
                            float(v) if isinstance(v, (np.floating, np.float64, np.float32)) else
                            v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in config.items()},
        "cell_frame_counts": {str(k): int(v) for k, v in cell_frame_count.items()}
    }
    
    metadata_file = output_path.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to: {metadata_file}")
    
    return Z, cell_id_array, frame_in_cell_array, abs_time_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export all frame latents (Step 1)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch_50.pt",
                       help="Path to checkpoint file")
    parser.add_argument("--index_csv", type=str, default="index.csv",
                       help="Path to index CSV file")
    parser.add_argument("--output", type=str, default="latents_all_frames.npz",
                       help="Output .npz file path")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing")
    parser.add_argument("--max_sequences", type=int, default=None,
                       help="Maximum number of sequences to process")
    
    args = parser.parse_args()
    
    export_all_frame_latents(
        checkpoint_path=args.checkpoint,
        index_csv=args.index_csv,
        output_file=args.output,
        batch_size=args.batch_size,
        max_sequences=args.max_sequences
    )

