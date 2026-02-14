"""
Generate Reconstruction Examples from Trained Model
- Loads trained checkpoint
- Selects sample sequences from dataset
- Generates reconstructions
- Saves comparison images (original vs reconstructed)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
from tqdm import tqdm

from model import ConvLSTMAutoencoder
from dataset_ivf import IVFSequenceDataset
import re


def save_reconstruction_comparison(
    original,  # [T, C, H, W]
    reconstructed,  # [T, C, H, W]
    cell_id,
    start_idx,
    output_path,
    n_frames_to_show=None  # None = show all frames
):
    """
    Save side-by-side comparison of original and reconstructed frames
    Uses PIL instead of matplotlib to avoid dependencies
    
    Args:
        original: Original frames [T, C, H, W]
        reconstructed: Reconstructed frames [T, C, H, W]
        cell_id: Cell ID for filename
        start_idx: Start index for filename
        output_path: Directory to save images
        n_frames_to_show: Number of frames to display (default: 10)
    """
    # Validate inputs
    if isinstance(original, torch.Tensor):
        original = original.numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.numpy()
    
    # Check shapes
    if len(original.shape) != 4 or len(reconstructed.shape) != 4:
        raise ValueError(f"Expected 4D tensors [T, C, H, W], got original: {original.shape}, reconstructed: {reconstructed.shape}")
    
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}")
    
    T = original.shape[0]
    if T == 0:
        raise ValueError(f"Empty sequence: T={T}")
    
    if n_frames_to_show is None or n_frames_to_show >= T:
        frame_indices = np.arange(T)
        n_frames_to_show = T
    else:
        frame_indices = np.linspace(0, T - 1, n_frames_to_show, dtype=int)
    
    # Get frame dimensions
    H, W = original.shape[2], original.shape[3]
    frame_size = 128  # Output frame size
    
    # Create canvas: 2 rows (original, reconstructed) x n_frames columns
    canvas_width = n_frames_to_show * frame_size
    canvas_height = 2 * frame_size + 60  # Extra space for labels
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw title
    title = f"Cell: {cell_id}, Start: {start_idx}"
    draw.text((canvas_width // 2 - len(title) * 3, 5), title, fill='black', font=font)
    
    for i, t_idx in enumerate(frame_indices):
        x_offset = i * frame_size
        
        # Original frame (already detached and on CPU)
        if isinstance(original, torch.Tensor):
            orig_frame = original[t_idx, 0].numpy()  # [H, W]
        else:
            orig_frame = original[t_idx, 0]  # Already numpy
        
        # Normalize to [0, 1] using percentile to handle outliers
        p1, p99 = np.percentile(orig_frame, [1, 99])
        if p99 > p1:
            orig_frame = np.clip((orig_frame - p1) / (p99 - p1 + 1e-8), 0, 1)
        else:
            orig_frame = np.clip(orig_frame, 0, 1)
        orig_frame = (orig_frame * 255).astype(np.uint8)
        orig_img = Image.fromarray(orig_frame, mode='L').resize((frame_size, frame_size), Image.BILINEAR)
        orig_img_rgb = orig_img.convert('RGB')
        canvas.paste(orig_img_rgb, (x_offset, 30))
        
        # Label for original
        label = f"Orig {t_idx}"
        draw.text((x_offset + 5, 30 + frame_size + 5), label, fill='black', font=font_small)
        
        # Reconstructed frame (already detached, clamped to [0,1], and on CPU)
        if isinstance(reconstructed, torch.Tensor):
            recon_frame = reconstructed[t_idx, 0].numpy()  # [H, W]
        else:
            recon_frame = reconstructed[t_idx, 0]  # Already numpy
        
        # Should already be in [0, 1] range from clamp, but ensure it
        recon_frame = np.clip(recon_frame, 0, 1)
        recon_frame = (recon_frame * 255).astype(np.uint8)
        recon_img = Image.fromarray(recon_frame, mode='L').resize((frame_size, frame_size), Image.BILINEAR)
        recon_img_rgb = recon_img.convert('RGB')
        canvas.paste(recon_img_rgb, (x_offset, 30 + frame_size + 25))
        
        # Label for reconstructed
        label = f"Recon {t_idx}"
        draw.text((x_offset + 5, 30 + frame_size + 25 + frame_size + 5), label, fill='black', font=font_small)
    
    # Save
    filename = f"reconstruction_{cell_id}_start{start_idx}.png"
    filepath = output_path / filename
    canvas.save(filepath, 'PNG')
    
    return filepath


def parse_sort_key(p: Path):
    """Parse sorting key from path name"""
    name = p.name
    run_pat = re.compile(r'RUN[_\- ]?(\d+)', re.I)
    num_pat = re.compile(r'(\d+)')
    run_m = run_pat.search(name)
    run_idx = int(run_m.group(1)) if run_m else 10**9
    nums = [int(x) for x in num_pat.findall(name)]
    nums = tuple(nums) if nums else ()
    mtime = p.stat().st_mtime_ns
    return (run_idx, nums, mtime)


def list_all_frames_in_cell(cell_dir: Path):
    """List ALL image frames in a cell directory (no subsampling)"""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    frames = []
    for ext in exts:
        frames += [Path(p) for p in cell_dir.glob(ext)]
    frames = [p for p in frames if p.exists() and p.stat().st_size > 0]
    frames.sort(key=parse_sort_key)
    return frames


def generate_reconstructions(
    checkpoint_path,
    index_csv="index.csv",
    output_dir="reconstructions",
    num_samples=10,
    batch_size=1,  # Use 1 to get individual sequences
    device="cuda" if torch.cuda.is_available() else "cpu",
    n_frames_per_sample=None,  # None = show all frames
    seq_len=20,
    use_all_frames=False,  # If True, read all frames from cell folder (no subsample)
    data_root="data",  # Root directory for cell folders (when use_all_frames=True)
    cell_id=None,  # Specific cell ID (when use_all_frames=True)
    max_frames=435  # Maximum number of frames to process (0-434, i.e., 435 frames)
):
    """
    Generate reconstruction examples from trained model
    
    Args:
        checkpoint_path: Path to checkpoint file
        index_csv: Path to index CSV file
        output_dir: Directory to save reconstruction images
        num_samples: Number of sample sequences to reconstruct
        batch_size: Batch size (use 1 for individual sequences)
        device: Device to use
        n_frames_per_sample: Number of frames to show per sample
        seq_len: Sequence length
    """
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load model
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    config = checkpoint.get('config', {})
    model_seq_len = config.get('seq_len', seq_len)
    
    # Initialize model
    model = ConvLSTMAutoencoder(
        seq_len=model_seq_len,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        use_classifier=False
    )
    
    # Load weights (use strict=False to handle model structure differences)
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model loaded successfully (with strict=False)!")
    except Exception as e:
        print(f"Warning: Could not load model weights: {e}")
        print("Attempting to load with manual key mapping...")
        # Try to map keys manually if needed
        state_dict = checkpoint['model_state_dict']
        model_state = model.state_dict()
        # Filter out keys that don't match
        filtered_state = {k: v for k, v in state_dict.items() if k in model_state}
        if len(filtered_state) > 0:
            model.load_state_dict(filtered_state, strict=False)
            print(f"Loaded {len(filtered_state)}/{len(state_dict)} weights")
        else:
            raise RuntimeError("Could not load any weights from checkpoint")
    
    model.to(device)
    model.eval()
    
    saved_files = []
    
    # Option 1: Read ALL frames directly from cell folder (no subsample)
    if use_all_frames:
        print(f"\n=== Mode: Read ALL frames from cell folder (no subsample) ===")
        data_root = Path(data_root)
        
        if cell_id is None:
            # Use first cell found
            cell_dirs = [p for p in data_root.iterdir() if p.is_dir()]
            if not cell_dirs:
                raise ValueError(f"No cell directories found in {data_root}")
            cell_dir = sorted(cell_dirs, key=lambda x: x.name)[0]
            cell_id = cell_dir.name
            print(f"No cell_id specified, using first cell: {cell_id}")
        else:
            cell_dir = data_root / cell_id
            if not cell_dir.exists():
                raise ValueError(f"Cell directory not found: {cell_dir}")
        
        print(f"Processing cell: {cell_id}")
        print(f"Cell directory: {cell_dir}")
        
        # List ALL frames (no subsampling)
        all_frames = list_all_frames_in_cell(cell_dir)
        print(f"Found {len(all_frames)} total frames (no subsample)")
        
        if len(all_frames) == 0:
            raise ValueError(f"No frames found in {cell_dir}")
        
        # Limit to max_frames (0 to max_frames-1)
        if len(all_frames) > max_frames:
            all_frames = all_frames[:max_frames]
            print(f"Limited to first {max_frames} frames (indices 0-{max_frames-1})")
        else:
            print(f"Processing all {len(all_frames)} frames (less than {max_frames})")
        
        # Load all frames as a sequence
        print("Loading all frames...")
        frames_tensors = []
        for frame_path in tqdm(all_frames, desc="Loading frames"):
            img = Image.open(frame_path)
            img = img.convert("L")
            img = img.resize((128, 128), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            # Normalize to [0, 1]
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
            frames_tensors.append(tensor)
        
        # Stack into sequence [T, 1, H, W]
        full_sequence = torch.stack(frames_tensors, dim=0)  # [T, 1, H, W]
        print(f"Full sequence shape: {full_sequence.shape}")
        
        # Process in chunks of seq_len (sliding window)
        print(f"\nProcessing with sliding windows of size {model_seq_len}...")
        T_total = full_sequence.shape[0]
        chunk_size = model_seq_len
        stride = chunk_size // 2  # 50% overlap
        
        print(f"Total frames: {T_total}, chunk_size: {chunk_size}, stride: {stride}")
        
        all_original = []
        all_reconstructed = []
        
        with torch.no_grad():
            window_starts = list(range(0, T_total - chunk_size + 1, stride))
            # Ensure we cover the last frames
            if window_starts[-1] + chunk_size < T_total:
                # Add one more window to cover remaining frames
                window_starts.append(max(0, T_total - chunk_size))
            
            print(f"Processing {len(window_starts)} windows...")
            
            for window_idx, start_idx in enumerate(tqdm(window_starts, desc="Reconstructing windows")):
                end_idx = min(start_idx + chunk_size, T_total)
                chunk = full_sequence[start_idx:end_idx]  # [actual_size, 1, H, W]
                
                # Pad if necessary (shouldn't happen, but just in case)
                if chunk.shape[0] < chunk_size:
                    padding = torch.zeros((chunk_size - chunk.shape[0], 1, chunk.shape[2], chunk.shape[3]))
                    chunk = torch.cat([chunk, padding], dim=0)
                
                chunk = chunk.unsqueeze(0).to(device)  # [1, chunk_size, 1, H, W]
                
                # Forward pass
                output = model(chunk)
                reconstruction = output['reconstruction']  # [1, chunk_size, 1, H, W]
                
                # Store (handle overlap by taking only non-overlapping parts)
                if start_idx == 0:
                    # First chunk: take all frames up to end_idx
                    actual_frames = min(chunk_size, T_total - start_idx)
                    all_original.append(chunk.squeeze(0)[:actual_frames].cpu().detach())
                    all_reconstructed.append(reconstruction.squeeze(0)[:actual_frames].cpu().detach().clamp(0, 1))
                else:
                    # Subsequent chunks: take only non-overlapping part (last half)
                    overlap = chunk_size - stride
                    actual_frames = min(chunk_size - overlap, T_total - start_idx)
                    if actual_frames > 0:
                        all_original.append(chunk.squeeze(0)[overlap:overlap+actual_frames].cpu().detach())
                        all_reconstructed.append(reconstruction.squeeze(0)[overlap:overlap+actual_frames].cpu().detach().clamp(0, 1))
        
        # Concatenate all chunks
        original_full = torch.cat(all_original, dim=0)  # [T', 1, H, W]
        reconstructed_full = torch.cat(all_reconstructed, dim=0)  # [T', 1, H, W]
        
        print(f"Final sequence shape: {original_full.shape}")
        T_total = original_full.shape[0]
        print(f"Total frames processed: {T_total}")
        
        # Save in chunks of 16 frames per image
        chunk_size = 16
        num_chunks = (T_total + chunk_size - 1) // chunk_size
        
        print(f"\nSaving {num_chunks} images, each with up to {chunk_size} frames...")
        print(f"Expected: {num_chunks} chunks for {T_total} frames")
        
        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * chunk_size
            end_frame = min(start_frame + chunk_size, T_total)
            
            if start_frame >= T_total:
                print(f"  ⚠️  Skipping chunk {chunk_idx+1}: start_frame {start_frame} >= T_total {T_total}")
                break
            
            original_chunk = original_full[start_frame:end_frame]  # [chunk_size, 1, H, W]
            reconstructed_chunk = reconstructed_full[start_frame:end_frame]  # [chunk_size, 1, H, W]
            
            print(f"  Processing chunk {chunk_idx+1}/{num_chunks}: frames {start_frame}-{end_frame-1} (shape: {original_chunk.shape})")
            
            try:
                filepath = save_reconstruction_comparison(
                    original_chunk,
                    reconstructed_chunk,
                    cell_id,
                    start_frame,  # start_idx = start_frame
                    output_path,
                    n_frames_to_show=end_frame - start_frame  # Show all frames in this chunk
                )
                saved_files.append(filepath)
                print(f"  ✓ [{chunk_idx+1}/{num_chunks}] Saved frames {start_frame}-{end_frame-1}: {filepath.name}")
            except Exception as e:
                print(f"  ✗ ERROR saving chunk {chunk_idx+1}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ Saved {len(saved_files)} reconstruction images!")
        print(f"Expected {num_chunks} images, got {len(saved_files)} images")
        if len(saved_files) != num_chunks:
            print(f"⚠️  WARNING: Expected {num_chunks} images but only saved {len(saved_files)}!")
    
    # Option 2: Use index.csv (original method with subsample)
    else:
        # Load dataset
        print(f"Loading dataset from: {index_csv}")
        dataset = IVFSequenceDataset(index_csv, resize=128, norm="minmax01")
        
        # Select random samples
        total_samples = len(dataset)
        num_samples = min(num_samples, total_samples)
        sample_indices = np.random.choice(total_samples, num_samples, replace=False)
        print(f"Selected {num_samples} random samples from {total_samples} total sequences")
        
        # Generate reconstructions
        print("\nGenerating reconstructions...")
        
        with torch.no_grad():
            for i, idx in enumerate(tqdm(sample_indices, desc="Reconstructing")):
                # Get single sequence (dataset returns tuple: (sequence, cell_id))
                sample = dataset[idx]
                if isinstance(sample, tuple):
                    sequence, cell_id = sample
                    # Get start_idx from dataset
                    start_idx = dataset.df.iloc[idx]["start_idx"] if hasattr(dataset, 'df') else 0
                else:
                    sequence = sample['sequence']
                    cell_id = sample['cell_id']
                    start_idx = sample.get('start_idx', 0)
                
                sequence = sequence.unsqueeze(0).to(device)  # [1, T, C, H, W]
                
                # Forward pass
                output = model(sequence)
                reconstruction = output['reconstruction']  # [1, T, C, H, W]
                
                # Debug: Check output immediately after model
                if i == 0:
                    print(f"\n[Debug] Model output check:")
                    print(f"  Output keys: {list(output.keys())}")
                    print(f"  Reconstruction shape: {reconstruction.shape}")
                    print(f"  Reconstruction min/max: {reconstruction.min().item():.6f} / {reconstruction.max().item():.6f}")
                    print(f"  Reconstruction has NaN: {torch.isnan(reconstruction).any().item()}")
                    print(f"  Reconstruction has Inf: {torch.isinf(reconstruction).any().item()}")
                
                # Remove batch dimension and detach from computation graph
                original = sequence.squeeze(0).cpu().detach()  # [T, C, H, W]
                reconstructed = reconstruction.squeeze(0).cpu().detach()  # [T, C, H, W]
                
                # Check for NaN/Inf before clamp
                if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
                    print(f"  ⚠️  WARNING: Reconstructed has NaN/Inf! Skipping this sample.")
                    continue
                
                # Clamp to [0, 1] range (model output should already be in this range after sigmoid)
                reconstructed = reconstructed.clamp(0, 1)
                
                # Debug: Check data ranges (only for first sample)
                if i == 0:
                    print(f"\n[Debug] After processing:")
                    print(f"  Original shape: {original.shape}, min={original.min():.4f}, max={original.max():.4f}, mean={original.mean():.4f}")
                    print(f"  Reconstructed shape: {reconstructed.shape}, min={reconstructed.min():.4f}, max={reconstructed.max():.4f}, mean={reconstructed.mean():.4f}")
                    print(f"  Reconstructed std: {reconstructed.std():.6f}")
                    
                    # Check if all values are the same (collapse to constant)
                    if reconstructed.std() < 0.01:
                        print(f"  ⚠️  WARNING: Reconstructed std is very small ({reconstructed.std():.6f}), model may have collapsed to constant!")
                
                # Validate before saving
                if original.shape != reconstructed.shape:
                    print(f"  ⚠️  WARNING: Shape mismatch! Original: {original.shape}, Reconstructed: {reconstructed.shape}")
                    continue
                
                if original.numel() == 0 or reconstructed.numel() == 0:
                    print(f"  ⚠️  WARNING: Empty tensors! Original: {original.numel()}, Reconstructed: {reconstructed.numel()}")
                    continue
                
                # Save comparison
                try:
                    n_frames_to_show = n_frames_per_sample if n_frames_per_sample is not None else original.shape[0]
                    filepath = save_reconstruction_comparison(
                        original,
                        reconstructed,
                        cell_id,
                        start_idx,
                        output_path,
                        n_frames_to_show=n_frames_to_show
                    )
                    saved_files.append(filepath)
                    print(f"  [{i+1}/{num_samples}] Saved: {filepath.name}")
                except Exception as e:
                    print(f"  ❌ ERROR saving {cell_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print(f"\n✅ Generated {len(saved_files)} reconstruction examples!")
    print(f"📁 Files saved in: {output_dir}/")
    print(f"\nSaved files:")
    for f in saved_files:
        print(f"  - {f.name}")
    
    return saved_files


def generate_single_sequence_reconstruction(
    checkpoint_path,
    index_csv="index.csv",
    cell_id=None,
    start_idx=None,
    output_dir="reconstructions",
    device="cuda" if torch.cuda.is_available() else "cpu",
    n_frames_to_show=None  # None = show all frames
):
    """
    Generate reconstruction for a specific sequence
    
    Args:
        checkpoint_path: Path to checkpoint file
        index_csv: Path to index CSV file
        cell_id: Specific cell ID to reconstruct (if None, random)
        start_idx: Specific start index (if None, random)
        output_dir: Directory to save reconstruction image
        device: Device to use
        n_frames_to_show: Number of frames to display
    """
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load model
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    model_seq_len = config.get('seq_len', 20)
    
    # Initialize model
    model = ConvLSTMAutoencoder(
        seq_len=model_seq_len,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        use_classifier=False
    )
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model loaded successfully (with strict=False)!")
    except Exception as e:
        print(f"Warning: Could not load model weights: {e}")
        state_dict = checkpoint['model_state_dict']
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in model_state}
        if len(filtered_state) > 0:
            model.load_state_dict(filtered_state, strict=False)
            print(f"Loaded {len(filtered_state)}/{len(state_dict)} weights")
        else:
            raise RuntimeError("Could not load any weights from checkpoint")
    
    model.to(device)
    model.eval()
    
    # Load dataset
    dataset = IVFSequenceDataset(index_csv, resize=128, norm="minmax01")
    
    # Find specific sequence or random
    if cell_id is not None and start_idx is not None:
        # Find specific sequence
        found = False
        for idx, sample in enumerate(dataset):
            if sample['cell_id'] == cell_id and sample['start_idx'] == start_idx:
                sequence_idx = idx
                found = True
                break
        if not found:
            print(f"⚠️  Sequence not found: cell_id={cell_id}, start_idx={start_idx}")
            return None
    else:
        # Random sequence
        sequence_idx = np.random.randint(len(dataset))
        sample = dataset[sequence_idx]
        cell_id = sample['cell_id']
        start_idx = sample['start_idx']
        print(f"Selected random sequence: cell_id={cell_id}, start_idx={start_idx}")
    
    # Get sequence (dataset returns tuple: (sequence, cell_id))
    sample = dataset[sequence_idx]
    if isinstance(sample, tuple):
        sequence, _ = sample
        start_idx = dataset.df.iloc[sequence_idx]["start_idx"] if hasattr(dataset, 'df') else 0
    else:
        sequence = sample['sequence']
        start_idx = sample.get('start_idx', 0)
    
    sequence = sequence.unsqueeze(0).to(device)  # [1, T, C, H, W]
    
    # Forward pass
    print("Generating reconstruction...")
    with torch.no_grad():
        output = model(sequence)
        reconstruction = output['reconstruction']  # [1, T, C, H, W]
    
    # Remove batch dimension and detach from computation graph
    original = sequence.squeeze(0).cpu().detach()  # [T, C, H, W]
    reconstructed = reconstruction.squeeze(0).cpu().detach()  # [T, C, H, W]
    
    # Clamp to [0, 1] range
    reconstructed = reconstructed.clamp(0, 1)
    
    if n_frames_to_show is None:
        n_frames_to_show = original.shape[0]
    filepath = save_reconstruction_comparison(
        original,
        reconstructed,
        cell_id,
        start_idx,
        output_path,
        n_frames_to_show=n_frames_to_show
    )
    
    print(f"✅ Saved reconstruction to: {filepath}")
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reconstruction examples from trained model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch_50.pt",
                       help="Path to checkpoint file")
    parser.add_argument("--index_csv", type=str, default="index.csv",
                       help="Path to index CSV file")
    parser.add_argument("--output_dir", type=str, default="reconstructions",
                       help="Output directory for reconstruction images")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of sample sequences to reconstruct")
    parser.add_argument("--n_frames", type=int, default=None,
                       help="Number of frames to show per sample (None = show all frames)")
    parser.add_argument("--cell_id", type=str, default=None,
                       help="Specific cell ID to reconstruct (optional)")
    parser.add_argument("--start_idx", type=int, default=None,
                       help="Specific start index to reconstruct (optional)")
    parser.add_argument("--use_all_frames", action="store_true",
                       help="Read ALL frames from cell folder (no subsample, bypass index.csv)")
    parser.add_argument("--data_root", type=str, default="data",
                       help="Root directory for cell folders (when use_all_frames=True)")
    parser.add_argument("--max_frames", type=int, default=435,
                       help="Maximum number of frames to process (default: 435, i.e., 0-434)")
    
    args = parser.parse_args()
    
    if args.cell_id is not None and args.start_idx is not None:
        # Generate single specific sequence
        generate_single_sequence_reconstruction(
            checkpoint_path=args.checkpoint,
            index_csv=args.index_csv,
            cell_id=args.cell_id,
            start_idx=args.start_idx,
            output_dir=args.output_dir,
            n_frames_to_show=args.n_frames
        )
    else:
        # Generate multiple random samples
        generate_reconstructions(
            checkpoint_path=args.checkpoint,
            index_csv=args.index_csv,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            n_frames_per_sample=args.n_frames,
            use_all_frames=args.use_all_frames,
            data_root=args.data_root,
            cell_id=args.cell_id,
            max_frames=args.max_frames
        )

