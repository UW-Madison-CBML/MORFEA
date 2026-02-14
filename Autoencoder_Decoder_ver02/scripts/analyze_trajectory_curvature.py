#!/usr/bin/env python3
"""
Analyze embryo trajectory curvature using circle-fitting method.

This script:
1. Loads latent vectors for a given video_name
2. Computes 3D TPHATE embedding
3. Calculates curvature along the trajectory
4. Visualizes trajectory colored by curvature
5. Identifies and extracts high-curvature frames
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
import sys
from PIL import Image
import tarfile
import io
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

# Add possible paths for imports
possible_paths = [
    str(Path(__file__).parent.parent),  # Parent of scripts/
    str(Path(__file__).parent.parent.parent / 'ivf_repo'),  # ~/ivf_repo
    str(Path.home() / 'ivf_repo'),  # ~/ivf_repo
    '/staging/groups/bhaskar_group/rho9/ivf_analysis',  # Group analysis dir
]

for path in possible_paths:
    if Path(path).exists():
        sys.path.insert(0, path)

try:
    from dataset_ivf import IVFSequenceDataset
except ImportError as e:
    print(f"Error importing dataset_ivf: {e}")
    print("\nTried paths:")
    for path in possible_paths:
        exists = "✓" if Path(path).exists() else "✗"
        print(f"  {exists} {path}")
    print("\nMake sure dataset_ivf.py is in one of these locations")
    sys.exit(1)

# Try to import model classes (handle different model structures)
try:
    from model import ConvLSTMAutoencoder
    ConvLSTMAutoencoder_AVAILABLE = True
except ImportError:
    ConvLSTMAutoencoder_AVAILABLE = False

try:
    from model import Encoder, Decoder
    EncoderDecoder_AVAILABLE = True
except ImportError:
    EncoderDecoder_AVAILABLE = False

try:
    from model import FrameEncoder, FrameDecoder
    FrameEncoderDecoder_AVAILABLE = True
except ImportError:
    FrameEncoderDecoder_AVAILABLE = False

if not (ConvLSTMAutoencoder_AVAILABLE or EncoderDecoder_AVAILABLE):
    print("Error: Could not import model classes")
    print("Available in model.py:")
    try:
        import model
        print(f"  Classes: {[x for x in dir(model) if not x.startswith('_')]}")
    except:
        pass
    sys.exit(1)

import torch.nn as nn

# Try to import tphate
try:
    import tphate
    TPHATE_AVAILABLE = True
except ImportError:
    TPHATE_AVAILABLE = False
    print("Warning: tphate not available, will use PHATE as fallback")
    try:
        import phate
        PHATE_AVAILABLE = True
    except ImportError:
        PHATE_AVAILABLE = False
        print("Error: Neither tphate nor phate available")
        sys.exit(1)


def load_model(checkpoint_path="checkpoints/checkpoint_epoch_50.pt", device="cpu"):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    if not Path(checkpoint_path).exists():
        # Try alternative paths
        alt_paths = [
            "checkpoint_epoch_50.pt",
            "checkpoints/checkpoint_epoch_50.pt",
            "../checkpoints/checkpoint_epoch_50.pt"
        ]
        for alt in alt_paths:
            if Path(alt).exists():
                checkpoint_path = alt
                break
        else:
            raise FileNotFoundError(f"Model checkpoint not found. Tried: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to determine model structure from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Check model structure by examining state_dict keys
    state_keys = list(state_dict.keys())
    
    # Try ConvLSTMAutoencoder first (most common)
    if ConvLSTMAutoencoder_AVAILABLE:
        try:
            # Determine model parameters from checkpoint
            # Check for encoder_hidden_dim (look for frame_encoder.proj.weight)
            if any('frame_encoder.proj.weight' in k for k in state_keys):
                # Extract dimension from weight shape
                for k in state_keys:
                    if 'frame_encoder.proj.weight' in k:
                        encoder_dim = state_dict[k].shape[0]
                        break
                else:
                    encoder_dim = 256  # default
            else:
                encoder_dim = 256  # default
            
            model = ConvLSTMAutoencoder(
                seq_len=20,
                encoder_hidden_dim=encoder_dim,
                encoder_layers=2,
                decoder_hidden_dim=128,
                decoder_layers=2
            )
            model.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded model (ConvLSTMAutoencoder, encoder_dim={encoder_dim})")
        except Exception as e:
            print(f"Warning: Failed to load as ConvLSTMAutoencoder: {e}")
            # Try Encoder + Decoder structure if available
            if EncoderDecoder_AVAILABLE:
                try:
                    encoder = Encoder()
                    decoder = Decoder()
                    class Autoencoder(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.encoder = encoder
                            self.decoder = decoder
                        
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
                    
                    model = Autoencoder()
                    model.load_state_dict(state_dict, strict=False)
                    print("✓ Loaded model (Encoder + Decoder structure)")
                except Exception as e2:
                    raise RuntimeError(f"Failed to load model with both structures: {e}, {e2}")
            else:
                raise
    else:
        # Only EncoderDecoder available
        if EncoderDecoder_AVAILABLE:
            encoder = Encoder()
            decoder = Decoder()
            class Autoencoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = encoder
                    self.decoder = decoder
                
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
            
            model = Autoencoder()
            model.load_state_dict(state_dict, strict=False)
            print("✓ Loaded model (Encoder + Decoder structure)")
        else:
            raise RuntimeError("No model classes available. Check model.py imports.")
    
    model.to(device)
    model.eval()
    return model


def _load_from_tar(tar_file, video_name, model, device, max_frames=None):
    """
    Load frames directly from tar.gz file without extracting.
    
    Args:
        tar_file: Path to tar.gz file
        video_name: Cell ID (e.g., "ZS435-5")
        model: Trained autoencoder model
        device: Device to run model on
        max_frames: Maximum number of frames to process (None = no limit, but stop at empty well)
    
    Returns:
        latents: numpy array of shape [T, 256] (or [T, 128])
        frame_paths: List of tar member names
    """
    print(f"  Loading from tar.gz: {tar_file}")
    
    # List all frames for this cell in tar.gz
    # Use getnames() instead of getmembers() for faster listing (only names, no metadata)
    cell_prefix = f"embryo_dataset/{video_name}/"
    exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    print(f"  Scanning tar.gz for frames (this may take a moment for large files)...")
    frame_members = []
    with tarfile.open(tar_file, 'r:gz') as tar:
        # Use getnames() which is faster than getmembers() (only reads names, not full metadata)
        all_names = tar.getnames()
        for name in all_names:
            if name.startswith(cell_prefix) and name.lower().endswith(exts):
                frame_members.append(name)
    
    if len(frame_members) == 0:
        raise ValueError(f"No frames found for {video_name} in {tar_file}")
    
    # Sort
    frame_members = sorted(frame_members)
    
    # Apply max_frames limit if specified
    if max_frames is not None and len(frame_members) > max_frames:
        frame_members = frame_members[:max_frames]
        print(f"  Limited to first {max_frames} frames")
    
    print(f"  Processing {len(frame_members)} frames from tar.gz...")
    
    # Extract latent vectors
    latents = []
    valid_frames = []
    
    model.eval()
    with tarfile.open(tar_file, 'r:gz') as tar:
        with torch.no_grad():
            for idx, frame_name in enumerate(frame_members):
                if (idx + 1) % 50 == 0 or (idx + 1) == len(frame_members):
                    print(f"    Processing frame {idx + 1}/{len(frame_members)}...", end='\r')
                try:
                    # Extract image from tar (more efficient method)
                    member = tar.getmember(frame_name)
                    img_data = tar.extractfile(member)
                    if img_data is None:
                        continue
                    
                    # Read all data at once to avoid slow seek operations
                    img_bytes = img_data.read()
                    img_data.close()  # Close immediately to free resources
                    
                    # Load and preprocess image
                    img = Image.open(io.BytesIO(img_bytes))
                    img = img.convert("L")
                    img_array_full = np.array(img, dtype=np.float32)
                    
                    # Check for empty well (low variance/range indicates blank image)
                    img_range = img_array_full.max() - img_array_full.min()
                    img_std = img_array_full.std()
                    
                    # Empty well detection: very low variance/range
                    if img_range < 10 or img_std < 2:
                        print(f"    Detected empty well at frame {idx + 1}, stopping...")
                        break  # Stop processing if empty well detected
                    
                    img = img.resize((128, 128), Image.BILINEAR)
                    img_array = np.array(img, dtype=np.float32)
                    
                    # Normalize (minmax01)
                    lo, hi = np.percentile(img_array, [1, 99])
                    img_array = (img_array - lo) / (hi - lo + 1e-6)
                    img_array = np.clip(img_array, 0, 1)
                    
                    # Convert to tensor: [1, 1, 128, 128]
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
                    
                    # Extract latent
                    if hasattr(model, 'encoder') and hasattr(model.encoder, 'frame_encoder'):
                        z = model.encoder.frame_encoder(img_tensor)
                    elif hasattr(model, 'frame_encoder'):
                        z = model.frame_encoder(img_tensor)
                    else:
                        seq = img_tensor.unsqueeze(0)
                        output = model(seq)
                        if isinstance(output, dict):
                            z = output['z_seq'].squeeze(0).squeeze(0)
                        else:
                            z = output.squeeze(0).squeeze(0)
                    
                    # Flatten if needed
                    if z.dim() > 1:
                        z = z.flatten()
                    
                    latents.append(z.cpu().numpy())
                    valid_frames.append(frame_name)
                    
                except Exception as e:
                    print(f"    Warning: Failed to process {frame_name}: {e}")
                    continue
    
    if len(latents) == 0:
        raise ValueError(f"No valid frames processed for {video_name}")
    
    latents = np.array(latents)
    print(f"  ✓ Extracted {len(latents)} latent vectors, shape: {latents.shape}")
    
    return latents, valid_frames


def load_latent_vectors_for_video(video_name, model, data_root="data", device="cpu", max_frames=None):
    """
    Load or compute latent vectors for a given video_name.
    Supports both extracted directories and tar.gz files.
    
    Args:
        video_name: Cell ID (e.g., "ZS435-5")
        model: Trained autoencoder model
        data_root: Root directory containing cell folders, or path to tar.gz
        device: Device to run model on
        max_frames: Maximum number of frames to process
    
    Returns:
        latents: numpy array of shape [T, 256] (or [T, 128] depending on model)
        frame_paths: List of frame file paths (or tar member names)
    """
    print(f"\nLoading latent vectors for {video_name}...")
    
    # FIRST: Check if data_root is a tar.gz file - handle this immediately
    data_root_str = str(data_root)
    if data_root_str.endswith('.tar.gz') or data_root_str.endswith('.tgz'):
        tar_file = Path(data_root)
        if not tar_file.exists():
            # Try group tar.gz
            group_tar = Path('/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz')
            if group_tar.exists():
                tar_file = group_tar
                print(f"  Using group tar.gz: {tar_file}")
            else:
                raise FileNotFoundError(f"Tar.gz file not found: {data_root}")
        # Directly load from tar and return
        print(f"  Loading from tar.gz: {tar_file}")
        return _load_from_tar(tar_file, video_name, model, device, max_frames)
    
    # SECOND: Handle directory-based data_root
    tar_file = None
    data_path = Path(data_root)
    if not data_path.exists():
        # Try alternative paths
        if Path('data').exists():
            data_path = Path('data')
        elif Path('/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset').exists():
            data_path = Path('/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset')
        elif Path('/staging/groups/bhaskar_group/ivf/embryo_dataset').exists():
            data_path = Path('/staging/groups/bhaskar_group/ivf/embryo_dataset')
        else:
            # Try group tar.gz as fallback
            group_tar = Path('/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz')
            if group_tar.exists():
                tar_file = group_tar
                print(f"  Directory not found, using group tar.gz: {tar_file}")
                return _load_from_tar(tar_file, video_name, model, device, max_frames)
            else:
                raise FileNotFoundError(f"Data root not found: {data_root}")
    
    # Check if cell directory exists
    cell_dir = data_path / video_name
    if not cell_dir.exists():
        # Try group tar.gz as fallback
        group_tar = Path('/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz')
        if group_tar.exists():
            tar_file = group_tar
            print(f"  Cell directory not found, using group tar.gz: {tar_file}")
            return _load_from_tar(tar_file, video_name, model, device, max_frames)
        else:
            # List available cell directories to help user
            available_cells = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
            error_msg = f"Cell directory not found: {cell_dir}\n"
            error_msg += f"Available cells ({len(available_cells)} total):\n"
            if len(available_cells) <= 20:
                error_msg += "\n".join(f"  - {cell}" for cell in available_cells)
            else:
                error_msg += "\n".join(f"  - {cell}" for cell in available_cells[:20])
                error_msg += f"\n  ... and {len(available_cells) - 20} more"
            raise FileNotFoundError(error_msg)
    
    # Load from directory (original code)
    print(f"  Found cell directory: {cell_dir}")
    
    # List all frames
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    frames = []
    for ext in exts:
        frames += list(cell_dir.glob(ext))
    frames = sorted([p for p in frames if p.exists() and p.stat().st_size > 0])
    
    if len(frames) == 0:
        raise ValueError(f"No frames found in {cell_dir}")
    
    # Limit to max_frames if specified
    if max_frames is not None and len(frames) > max_frames:
        frames = frames[:max_frames]
        print(f"  Limited to first {max_frames} frames")
    
    print(f"  Processing {len(frames)} frames...")
    
    # Extract latent vectors
    latents = []
    valid_frames = []
    
    model.eval()
    with torch.no_grad():
        for frame_path in frames:
            try:
                # Load and preprocess image
                img = Image.open(frame_path)
                img = img.convert("L")
                img = img.resize((128, 128), Image.BILINEAR)
                img_array = np.array(img, dtype=np.float32)
                
                # Normalize (minmax01)
                lo, hi = np.percentile(img_array, [1, 99])
                img_array = (img_array - lo) / (hi - lo + 1e-6)
                img_array = np.clip(img_array, 0, 1)
                
                # Convert to tensor: [1, 1, 128, 128]
                img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
                
                # Extract latent
                if hasattr(model, 'encoder') and hasattr(model.encoder, 'frame_encoder'):
                    # New structure: Encoder + Decoder
                    z = model.encoder.frame_encoder(img_tensor)
                elif hasattr(model, 'frame_encoder'):
                    # ConvLSTMAutoencoder structure
                    z = model.frame_encoder(img_tensor)
                else:
                    # Use full model (expects sequence)
                    seq = img_tensor.unsqueeze(0)  # [1, 1, 1, 128, 128]
                    output = model(seq)
                    if isinstance(output, dict):
                        z = output['z_seq'].squeeze(0).squeeze(0)  # [256] or [128]
                    else:
                        z = output.squeeze(0).squeeze(0)
                
                z = z.cpu().numpy().flatten()
                latents.append(z)
                valid_frames.append(frame_path)
                
            except Exception as e:
                print(f"  Warning: Failed to process {frame_path}: {e}")
                continue
    
    if len(latents) == 0:
        raise ValueError(f"No valid frames processed for {video_name}")
    
    latents = np.array(latents)  # [T, latent_dim]
    print(f"  ✓ Extracted {len(latents)} latent vectors, shape: {latents.shape}")
    
    return latents, valid_frames


def compute_pca_3d(latents, n_components=3, device='cpu'):
    """
    Compute 3D PCA embedding from latent vectors.
    Can use GPU for faster computation on large datasets.
    
    Args:
        latents: numpy array of shape [T, latent_dim]
        n_components: Number of components (default: 3)
        device: 'cpu' or 'cuda' for GPU acceleration
    
    Returns:
        pca_embedding: numpy array of shape [T, 3]
    """
    print(f"\nComputing 3D PCA embedding...")
    print(f"  Input shape: {latents.shape}")
    print(f"  Requested device: {device}")
    
    # Check GPU availability
    use_gpu = (device == 'cuda' and torch.cuda.is_available())
    if device == 'cuda' and not torch.cuda.is_available():
        print(f"  ⚠ GPU requested but not available, using CPU instead")
        use_gpu = False
    
    try:
        if use_gpu:
            # GPU-accelerated PCA using PyTorch
            latents_tensor = torch.from_numpy(latents).float().to(device)
            
            # Standardize
            mean = latents_tensor.mean(dim=0)
            std = latents_tensor.std(dim=0)
            latents_scaled = (latents_tensor - mean) / (std + 1e-8)
            
            # SVD for PCA
            U, S, V = torch.linalg.svd(latents_scaled, full_matrices=False)
            
            # Get top n_components
            embedding = (U[:, :n_components] * S[:n_components]).cpu().numpy()
            
            # Calculate explained variance
            explained_variance = (S[:n_components] ** 2) / (S ** 2).sum()
            explained_variance = explained_variance.cpu().numpy()
            
            print(f"  ✓ PCA embedding shape: {embedding.shape} (GPU-accelerated)")
            print(f"  Explained variance ratio: {explained_variance}")
            print(f"  Total explained variance: {explained_variance.sum():.4f}")
        else:
            # CPU-based PCA using sklearn (more stable for small datasets)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            latents_scaled = scaler.fit_transform(latents)
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            embedding = pca.fit_transform(latents_scaled)
            
            print(f"  ✓ PCA embedding shape: {embedding.shape}")
            print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
            print(f"  Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        
        return embedding
        
    except Exception as e:
        raise RuntimeError(f"PCA failed: {e}")


def compute_tphate_3d(latents, n_components=3, knn=5):
    """
    Compute 3D TPHATE embedding from latent vectors.
    
    Args:
        latents: numpy array of shape [T, latent_dim]
        n_components: Number of dimensions (should be 3)
        knn: Number of nearest neighbors
    
    Returns:
        tphate_embedding: numpy array of shape [T, 3]
    """
    print(f"\nComputing 3D TPHATE embedding...")
    print(f"  Input shape: {latents.shape}")
    
    # TPHATE ONLY - no fallback to PHATE
    if not TPHATE_AVAILABLE:
        raise RuntimeError("TPHATE is REQUIRED but not available. Please install tphate.")
    
    print("Calculating TPHATE...")
    try:
        # Create TPHATE instance
        # TPHATE automatically infers temporal structure from data order
        tph = tphate.TPHATE(n_components=n_components, knn=knn, verbose=1)
        
        # Fit and transform (TPHATE infers temporal structure from data order)
        embedding = tph.fit_transform(latents)
        
        print(f"  ✓ TPHATE embedding shape: {embedding.shape}")
        return embedding
        
    except Exception as e:
        error_msg = f"TPHATE failed: {e}\nTPHATE is REQUIRED. No PHATE fallback allowed.\nPlease check tphate installation and try again."
        raise RuntimeError(error_msg)


def compute_curvature(trajectory):
    """
    Compute curvature along 3D trajectory using circle-fitting method.
    
    For each point p, uses triplet (p_prev, p, p_next) to compute curvature.
    
    Args:
        trajectory: numpy array of shape [T, 3] (3D points)
    
    Returns:
        curvatures: numpy array of shape [T] (curvature values)
    """
    print("\nComputing curvature...")
    
    T = trajectory.shape[0]
    curvatures = np.zeros(T)
    
    for i in range(1, T-1):
        p_prev = trajectory[i-1]
        p = trajectory[i]
        p_next = trajectory[i+1]
        
        # Compute side lengths
        a = np.linalg.norm(p - p_prev)
        b = np.linalg.norm(p_next - p)
        c = np.linalg.norm(p_next - p_prev)
        
        # Heron's formula for triangle area
        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)
        area = np.sqrt(np.maximum(area_sq, 0))
        
        # Compute curvature: kappa = 4 * area / (a * b * c)
        if area > 0 and a > 0 and b > 0 and c > 0:
            kappa = 4 * area / (a * b * c)
            curvatures[i] = kappa
        else:
            curvatures[i] = 0
    
    # First and last points: set to 0 or copy neighbors
    curvatures[0] = curvatures[1] if T > 1 else 0
    curvatures[-1] = curvatures[-2] if T > 1 else 0
    
    print(f"  ✓ Computed curvature for {T} points")
    print(f"    Min: {curvatures.min():.6f}, Max: {curvatures.max():.6f}, Mean: {curvatures.mean():.6f}")
    
    return curvatures


def plot_trajectory_curvature(trajectory, curvatures, video_name, save_path, method_name='TPHATE'):
    """
    Plot 3D trajectory colored by curvature.
    
    Args:
        trajectory: numpy array of shape [T, 3]
        curvatures: numpy array of shape [T]
        video_name: Name of the video (for title)
        save_path: Path to save the figure
        method_name: Name of the dimensionality reduction method (default: 'TPHATE')
    """
    print(f"\nPlotting trajectory colored by curvature...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot colored by curvature
    scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                        c=curvatures, cmap='viridis', s=50, alpha=0.7)
    
    # Add trajectory line
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
           'k-', alpha=0.3, linewidth=1)
    
    # Mark start and end
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
              c='green', s=200, marker='o', label='Start', edgecolors='black')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
              c='red', s=200, marker='s', label='End', edgecolors='black')
    
    ax.set_xlabel(f'{method_name} Component 1')
    ax.set_ylabel(f'{method_name} Component 2')
    ax.set_zlabel(f'{method_name} Component 3')
    ax.set_title(f'{method_name} trajectory (3D) colored by curvature – {video_name}')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Curvature', rotation=270, labelpad=20)
    
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved plot to {save_path}")


def extract_high_curvature_frames(frame_paths, high_curvature_indices, video_name, output_dir, tar_file=None):
    """
    Extract and save high-curvature frames.
    
    Args:
        frame_paths: List of frame file paths or tar member names
        high_curvature_indices: Array of indices where curvature is high
        video_name: Name of the video
        output_dir: Directory to save frames
        tar_file: Path to tar.gz file (if frame_paths are tar member names)
    """
    print(f"\nExtracting {len(high_curvature_indices)} high-curvature frames...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_frames = []
    
    # Check if frame_paths are tar member names (start with "embryo_dataset/")
    is_tar_members = tar_file is not None and len(frame_paths) > 0 and isinstance(frame_paths[0], str) and frame_paths[0].startswith("embryo_dataset/")
    
    if is_tar_members:
        # Load from tar.gz
        with tarfile.open(tar_file, 'r:gz') as tar:
            for idx in high_curvature_indices:
                if idx < len(frame_paths):
                    frame_member = frame_paths[idx]
                    output_path = output_dir / f"{video_name}_t{idx}.png"
                    
                    try:
                        member = tar.getmember(frame_member)
                        img_data = tar.extractfile(member)
                        if img_data is not None:
                            img = Image.open(io.BytesIO(img_data.read()))
                            img.save(output_path)
                            saved_frames.append(output_path)
                            img_data.close()
                    except Exception as e:
                        print(f"    Warning: Failed to extract {frame_member}: {e}")
                        continue
    else:
        # Load from regular file paths
        for idx in high_curvature_indices:
            if idx < len(frame_paths):
                frame_path = Path(frame_paths[idx])
                output_path = output_dir / f"{video_name}_t{idx}.png"
                
                try:
                    img = Image.open(frame_path)
                    img.save(output_path)
                    saved_frames.append(output_path)
                except Exception as e:
                    print(f"    Warning: Failed to load {frame_path}: {e}")
                    continue
    
    print(f"  ✓ Saved {len(saved_frames)} frames to {output_dir}")
    return saved_frames


def create_curvature_montage(frame_paths, high_curvature_indices, video_name, save_path, max_frames=16, tar_file=None):
    """
    Create a montage image showing all high-curvature frames in a grid.
    
    Args:
        frame_paths: List of frame file paths or tar member names
        high_curvature_indices: Array of indices where curvature is high
        video_name: Name of the video
        save_path: Path to save montage
        max_frames: Maximum number of frames to show in montage
        tar_file: Path to tar.gz file (if frame_paths are tar member names)
    """
    print(f"\nCreating curvature montage...")
    
    # Limit number of frames
    indices_to_show = high_curvature_indices[:max_frames]
    
    if len(indices_to_show) == 0:
        print("  No high-curvature frames to show")
        return
    
    # Calculate grid size
    n_frames = len(indices_to_show)
    cols = int(np.ceil(np.sqrt(n_frames)))
    rows = int(np.ceil(n_frames / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Check if frame_paths are tar member names
    is_tar_members = tar_file is not None and len(frame_paths) > 0 and isinstance(frame_paths[0], str) and frame_paths[0].startswith("embryo_dataset/")
    
    if is_tar_members:
        # Load from tar.gz
        with tarfile.open(tar_file, 'r:gz') as tar:
            for i, idx in enumerate(indices_to_show):
                if idx < len(frame_paths):
                    frame_member = frame_paths[idx]
                    try:
                        member = tar.getmember(frame_member)
                        img_data = tar.extractfile(member)
                        if img_data is not None:
                            img = Image.open(io.BytesIO(img_data.read()))
                            axes[i].imshow(img, cmap='gray')
                            axes[i].set_title(f'Frame {idx}', fontsize=8)
                            axes[i].axis('off')
                            img_data.close()
                    except Exception as e:
                        print(f"    Warning: Failed to load {frame_member}: {e}")
                        axes[i].axis('off')
    else:
        # Load from regular file paths
        for i, idx in enumerate(indices_to_show):
            if idx < len(frame_paths):
                try:
                    img = Image.open(frame_paths[idx])
                    axes[i].imshow(img, cmap='gray')
                    axes[i].set_title(f'Frame {idx}', fontsize=8)
                    axes[i].axis('off')
                except Exception as e:
                    print(f"    Warning: Failed to load {frame_paths[idx]}: {e}")
                    axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_frames, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'High-Curvature Frames – {video_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved montage to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze trajectory curvature')
    parser.add_argument('--video_name', type=str, required=True,
                       help='Video name (cell ID), e.g., ZS435-5')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: auto-detect in group directory)')
    parser.add_argument('--data_root', type=str, default=None,
                       help='Root directory containing cell folders (default: auto-detect in group directory)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: group directory or ./curvature_analysis)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run model on')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process (None = no limit, but stops at empty well)')
    parser.add_argument('--curvature_threshold_percentile', type=float, default=95.0,
                       help='Percentile threshold for high curvature (default: 95.0). Use None to disable percentile-based selection.')
    parser.add_argument('--curvature_threshold_absolute', type=float, default=None,
                       help='Absolute threshold for high curvature (default: None). If set, overrides percentile-based selection.')
    parser.add_argument('--curvature_max_frames', type=int, default=None,
                       help='Maximum number of high-curvature frames to extract (default: None = no limit)')
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'tphate'],
                       help='Dimensionality reduction method: pca or tphate (default: pca)')
    parser.add_argument('--knn', type=int, default=5,
                       help='Number of nearest neighbors for TPHATE (only used if method=tphate)')
    
    args = parser.parse_args()
    
    # Auto-detect paths in group directory
    GROUP_BASE = Path('/staging/groups/bhaskar_group/rho9')
    
    # Determine checkpoint path
    if args.checkpoint is None:
        # Try group directory first
        group_checkpoint = GROUP_BASE / 'checkpoints' / 'checkpoint_epoch_50.pt'
        home_checkpoint = Path.home() / 'ivf_repo' / 'checkpoints' / 'checkpoint_epoch_50.pt'
        local_checkpoint = Path('checkpoints/checkpoint_epoch_50.pt')
        
        if group_checkpoint.exists():
            args.checkpoint = str(group_checkpoint)
            print(f"Using group checkpoint: {args.checkpoint}")
        elif home_checkpoint.exists():
            args.checkpoint = str(home_checkpoint)
            print(f"Using home checkpoint: {args.checkpoint}")
        elif local_checkpoint.exists():
            args.checkpoint = str(local_checkpoint)
            print(f"Using local checkpoint: {args.checkpoint}")
        else:
            raise FileNotFoundError("Could not find checkpoint. Specify with --checkpoint")
    
    # Determine data root
    if args.data_root is None:
        # Try group directory first
        group_data = GROUP_BASE / 'ivf_data' / 'embryo_dataset'
        staging_data = Path('/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset')
        local_data = Path('data')
        group_tar = Path('/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz')
        
        if group_data.exists():
            args.data_root = str(group_data)
            print(f"Using group data: {args.data_root}")
        elif staging_data.exists():
            args.data_root = str(staging_data)
            print(f"Using staging data: {args.data_root}")
        elif local_data.exists():
            args.data_root = str(local_data)
            print(f"Using local data: {args.data_root}")
        elif group_tar.exists():
            # Use tar.gz as fallback
            args.data_root = str(group_tar)
            print(f"Using group tar.gz (no extraction needed): {args.data_root}")
        else:
            raise FileNotFoundError("Could not find data directory or tar.gz. Specify with --data_root")
    
    # Determine output directory (prefer group directory if available)
    if args.output_dir is None:
        # Try group directory first
        group_output = GROUP_BASE / 'curvature_analysis'
        if group_output.parent.exists():
            output_dir = group_output
            print(f"Using group output directory: {output_dir}")
        else:
            output_dir = Path('curvature_analysis')
            print(f"Using local output directory: {output_dir}")
    else:
        output_dir = Path(args.output_dir)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'frames' / 'high_curvature').mkdir(parents=True, exist_ok=True)
    
    # Verify GPU availability
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("⚠️  WARNING: CUDA requested but not available. Falling back to CPU.")
            args.device = 'cpu'
        else:
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("=" * 60)
    print("Trajectory Curvature Analysis")
    print("=" * 60)
    print(f"Video: {args.video_name}")
    print(f"Method: {args.method.upper()}")
    print(f"Device: {args.device}")
    if args.device == 'cuda' and torch.cuda.is_available():
        print(f"  ✓ Using GPU acceleration")
    else:
        print(f"  ⚠ Using CPU (slower)")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Load latent vectors
    latents, frame_paths = load_latent_vectors_for_video(
        args.video_name, model, args.data_root, args.device, args.max_frames
    )
    
    # Determine if data_root is a tar.gz file
    tar_file = None
    if str(args.data_root).endswith('.tar.gz') or str(args.data_root).endswith('.tgz'):
        tar_file = Path(args.data_root)
        if not tar_file.exists():
            # Try group tar.gz
            group_tar = Path('/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz')
            if group_tar.exists():
                tar_file = group_tar
    
    # Compute 3D embedding (PCA or TPHATE)
    print(f"\n{'='*60}")
    print(f"Step: Computing 3D {args.method.upper()} embedding")
    print(f"{'='*60}")
    
    # Verify method selection
    if args.method not in ['pca', 'tphate']:
        raise ValueError(f"Invalid method: {args.method}. Must be 'pca' or 'tphate'")
    
    if args.method == 'pca':
        print(f"✓ Using PCA for dimensionality reduction")
        trajectory = compute_pca_3d(latents, n_components=3, device=args.device)
        method_name = 'PCA'
    else:  # tphate
        print(f"✓ Using TPHATE for dimensionality reduction")
        trajectory = compute_tphate_3d(latents, n_components=3, knn=args.knn)
        method_name = 'TPHATE'
    
    print(f"\n✓ Completed: {method_name} embedding computed")
    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  Output file will be: {args.method}_curvature_{args.video_name}.png")
    
    # Compute curvature
    curvatures = compute_curvature(trajectory)
    
    # Plot trajectory colored by curvature
    plot_path = output_dir / 'figures' / f'{args.method}_curvature_{args.video_name}.png'
    plot_trajectory_curvature(trajectory, curvatures, args.video_name, plot_path, method_name=method_name)
    
    # Identify high-curvature regions
    if args.curvature_threshold_absolute is not None:
        # Use absolute threshold
        threshold = args.curvature_threshold_absolute
        high_curvature_indices = np.where(curvatures >= threshold)[0]
        print(f"\nHigh-curvature analysis (absolute threshold):")
        print(f"  Absolute threshold: {threshold:.6f}")
    else:
        # Use percentile-based threshold
        threshold = np.percentile(curvatures, args.curvature_threshold_percentile)
        high_curvature_indices = np.where(curvatures >= threshold)[0]
        print(f"\nHigh-curvature analysis (percentile-based):")
        print(f"  Threshold ({args.curvature_threshold_percentile}th percentile): {threshold:.6f}")
    
    # Also identify local peaks (curvature maxima) for additional important points
    peaks, _ = find_peaks(curvatures, height=threshold, distance=3)
    
    # Combine percentile/absolute-based and peak-based methods
    all_high_indices = np.unique(np.concatenate([high_curvature_indices, peaks]))
    all_high_indices = np.sort(all_high_indices)
    
    # Limit to max_frames if specified
    if args.curvature_max_frames is not None and len(all_high_indices) > args.curvature_max_frames:
        # Select top N by curvature value
        top_indices = np.argsort(curvatures[all_high_indices])[-args.curvature_max_frames:]
        all_high_indices = all_high_indices[top_indices]
        all_high_indices = np.sort(all_high_indices)
        print(f"  Limited to top {args.curvature_max_frames} frames by curvature value")
    
    total_frames = len(curvatures)
    expected_5_percent = int(total_frames * (100 - args.curvature_threshold_percentile) / 100)
    
    print(f"  Total frames: {total_frames}")
    print(f"  Expected top {(100 - args.curvature_threshold_percentile):.1f}%: ~{expected_5_percent} frames")
    print(f"  Threshold-based timepoints: {len(high_curvature_indices)}")
    print(f"  Local peaks (above threshold): {len(peaks)}")
    print(f"  Combined high-curvature timepoints: {len(all_high_indices)}")
    print(f"  Indices: {all_high_indices[:10]}..." if len(all_high_indices) > 10 else f"  Indices: {all_high_indices}")
    
    # Use combined indices
    high_curvature_indices = all_high_indices
    
    # Extract high-curvature frames
    frames_dir = output_dir / 'frames' / 'high_curvature'
    saved_frames = extract_high_curvature_frames(
        frame_paths, high_curvature_indices, args.video_name, frames_dir, tar_file=tar_file
    )
    
    # Create montage
    montage_path = output_dir / 'figures' / f'high_curvature_montage_{args.video_name}.png'
    create_curvature_montage(
        frame_paths, high_curvature_indices, args.video_name, montage_path, tar_file=tar_file
    )
    
    # Save curvature data (include method name to avoid overwriting)
    curvature_data_path = output_dir / f'curvature_data_{args.method}_{args.video_name}.npz'
    np.savez(curvature_data_path,
             method=args.method,
             trajectory=trajectory,
             curvatures=curvatures,
             high_curvature_indices=high_curvature_indices,
             threshold=threshold,
             max_curvature=np.max(curvatures),
             max_curvature_index=np.argmax(curvatures))
    print(f"\n✓ Saved curvature data to {curvature_data_path}")
    print(f"  Max curvature: {np.max(curvatures):.6f} at frame {np.argmax(curvatures)}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"  - Trajectory plot: {plot_path}")
    print(f"  - Montage: {montage_path}")
    print(f"  - High-curvature frames: {frames_dir}")
    print(f"  - Curvature data: {curvature_data_path}")


if __name__ == '__main__':
    main()

