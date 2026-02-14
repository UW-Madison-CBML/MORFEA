"""
Generate Reconstruction for ALL Original Frames (No Subsampling)
- Loads trained checkpoint
- Reads ALL frames directly from cell folder (bypassing index.csv subsampling)
- Generates reconstructions for every frame
- Saves comparison images
"""
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import re

from model import ConvLSTMAutoencoder


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


def list_all_frames(cell_dir: Path):
    """List ALL image frames in a cell directory (no subsampling)"""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    frames = []
    for ext in exts:
        frames += [Path(p) for p in cell_dir.glob(ext)]
    frames = [p for p in frames if p.exists() and p.stat().st_size > 0]
    frames.sort(key=parse_sort_key)
    return frames


def load_frame_as_tensor(frame_path: Path, resize=128):
    """Load a single frame as tensor"""
    img = Image.open(frame_path)
    img = img.convert("L")
    img = img.resize((resize, resize), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    # Normalize to [0, 1]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    # Convert to tensor [1, 1, H, W]
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def extract_frame_latent_from_encoder(model, frame_tensor, device):
    """Extract latent for a single frame using encoder"""
    with torch.no_grad():
        # Encoder expects [B, T, C, H, W], so we add T dimension
        frame_seq = frame_tensor.unsqueeze(1)  # [1, 1, 1, H, W]
        # Get encoder output
        encoded = model.frame_encoder(frame_seq)  # [1, 1, C, H', W']
        # Global average pooling to get latent vector
        B, T, C, H, W = encoded.shape
        latent = encoded.view(B, T, C, -1).mean(dim=-1)  # [1, 1, C]
        return latent.squeeze(0).squeeze(0).cpu().numpy()  # [C]


def reconstruct_frame(model, frame_tensor, device, seq_len=16):
    """
    Reconstruct a single frame by:
    1. Creating a sequence of the same frame repeated seq_len times
    2. Running through the full model
    3. Taking the last frame of reconstruction
    """
    with torch.no_grad():
        # Repeat frame to create sequence [1, seq_len, 1, H, W]
        frame_seq = frame_tensor.repeat(1, seq_len, 1, 1)
        frame_seq = frame_seq.to(device)
        
        # Forward pass
        output = model(frame_seq)
        reconstruction = output['reconstruction']  # [1, seq_len, 1, H, W]
        
        # Take the last frame (most refined)
        recon_frame = reconstruction[0, -1, 0].cpu().detach()  # [H, W]
        recon_frame = recon_frame.clamp(0, 1)
        
        return recon_frame.numpy()


def save_all_frames_comparison(
    original_frames,  # List of [H, W] arrays
    reconstructed_frames,  # List of [H, W] arrays
    cell_id,
    output_path,
    frames_per_row=20,
    frame_size=64
):
    """Save comparison of all original vs reconstructed frames"""
    n_frames = len(original_frames)
    n_rows = (n_frames + frames_per_row - 1) // frames_per_row
    
    canvas_width = frames_per_row * frame_size * 2  # 2 columns (original, reconstructed)
    canvas_height = n_rows * frame_size + 60  # Extra space for labels
    
    from PIL import ImageDraw, ImageFont
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Title
    title = f"Cell: {cell_id} - All {n_frames} Frames (Original vs Reconstructed)"
    draw.text((canvas_width // 2 - len(title) * 3, 5), title, fill='black', font=font)
    
    for i, (orig, recon) in enumerate(zip(original_frames, reconstructed_frames)):
        row = i // frames_per_row
        col = i % frames_per_row
        
        x_orig = col * frame_size * 2
        x_recon = x_orig + frame_size
        y = 30 + row * frame_size
        
        # Original frame
        orig_img = Image.fromarray((orig * 255).astype(np.uint8), mode='L')
        orig_img = orig_img.resize((frame_size, frame_size), Image.BILINEAR)
        canvas.paste(orig_img.convert('RGB'), (x_orig, y))
        
        # Reconstructed frame
        recon_img = Image.fromarray((recon * 255).astype(np.uint8), mode='L')
        recon_img = recon_img.resize((frame_size, frame_size), Image.BILINEAR)
        canvas.paste(recon_img.convert('RGB'), (x_recon, y))
        
        # Frame number label
        label = f"F{i}"
        draw.text((x_orig + 2, y + frame_size - 15), label, fill='black', font=font_small)
    
    # Save
    filename = f"reconstruction_all_frames_{cell_id}.png"
    filepath = output_path / filename
    canvas.save(filepath, 'PNG')
    return filepath


def generate_all_frames_reconstruction(
    checkpoint_path,
    data_root="data",
    cell_id=None,
    output_dir="reconstructions_all_frames",
    device="cuda" if torch.cuda.is_available() else "cpu",
    seq_len=16,
    max_frames=435  # Limit to first 435 frames (0-434)
):
    """
    Generate reconstruction for ALL original frames (no subsampling)
    
    Args:
        checkpoint_path: Path to checkpoint file
        data_root: Root directory containing cell folders
        cell_id: Specific cell ID (if None, process first cell found)
        output_dir: Output directory
        device: Device to use
        seq_len: Sequence length for model
        max_frames: Maximum number of frames to process
    """
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load model
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    model_seq_len = config.get('seq_len', seq_len)
    
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
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model weights: {e}")
        raise
    
    model.to(device)
    model.eval()
    
    # Find cell directory
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
    all_frames = list_all_frames(cell_dir)
    print(f"Found {len(all_frames)} total frames")
    
    # Limit to max_frames
    if len(all_frames) > max_frames:
        all_frames = all_frames[:max_frames]
        print(f"Limited to first {max_frames} frames (0-{max_frames-1})")
    
    print(f"Processing {len(all_frames)} frames...")
    
    # Process frames
    original_frames = []
    reconstructed_frames = []
    
    for i, frame_path in enumerate(tqdm(all_frames, desc="Reconstructing")):
        # Load frame
        frame_tensor = load_frame_as_tensor(frame_path, resize=128)
        frame_array = frame_tensor.squeeze().numpy()
        original_frames.append(frame_array)
        
        # Reconstruct
        recon_array = reconstruct_frame(model, frame_tensor, device, seq_len=model_seq_len)
        reconstructed_frames.append(recon_array)
    
    # Save comparison
    print(f"\nSaving comparison image...")
    filepath = save_all_frames_comparison(
        original_frames,
        reconstructed_frames,
        cell_id,
        output_path,
        frames_per_row=20,
        frame_size=64
    )
    
    print(f"✅ Saved to: {filepath}")
    print(f"   Total frames: {len(original_frames)}")
    
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reconstruction for ALL original frames (no subsampling)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch_50.pt",
                       help="Path to checkpoint file")
    parser.add_argument("--data_root", type=str, default="data",
                       help="Root directory containing cell folders")
    parser.add_argument("--cell_id", type=str, default=None,
                       help="Specific cell ID (if None, use first cell found)")
    parser.add_argument("--output_dir", type=str, default="reconstructions_all_frames",
                       help="Output directory")
    parser.add_argument("--max_frames", type=int, default=435,
                       help="Maximum number of frames to process (default: 435, i.e., 0-434)")
    
    args = parser.parse_args()
    
    generate_all_frames_reconstruction(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        cell_id=args.cell_id,
        output_dir=args.output_dir,
        max_frames=args.max_frames
    )

