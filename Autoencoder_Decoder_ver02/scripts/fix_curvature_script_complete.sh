#!/bin/bash
# 完整修復 analyze_trajectory_curvature.py 的腳本

cd /staging/groups/bhaskar_group/rho9/ivf_analysis/scripts

# 備份
cp analyze_trajectory_curvature.py analyze_trajectory_curvature.py.bak

# 使用 Python 完整替換 load_latent_vectors_for_video 函數
python3 << 'PYEOF'
import re

with open('analyze_trajectory_curvature.py', 'r') as f:
    content = f.read()

# 找到函數開始和結束
func_start = content.find('def load_latent_vectors_for_video(')
if func_start == -1:
    print("❌ 找不到函數")
    exit(1)

# 找到下一個函數定義（作為結束標記）
next_func = content.find('\ndef ', func_start + 1)
if next_func == -1:
    next_func = len(content)

# 提取函數前的內容和函數後的內容
before_func = content[:func_start]
after_func = content[next_func:]

# 新的完整函數（從本地文件複製的完整版本）
new_function = '''def load_latent_vectors_for_video(video_name, model, data_root="data", device="cpu", max_frames=435):
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
    print(f"\\nLoading latent vectors for {video_name}...")
    
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
            error_msg = f"Cell directory not found: {cell_dir}\\n"
            error_msg += f"Available cells ({len(available_cells)} total):\\n"
            if len(available_cells) <= 20:
                error_msg += "\\n".join(f"  - {cell}" for cell in available_cells)
            else:
                error_msg += "\\n".join(f"  - {cell}" for cell in available_cells[:20])
                error_msg += f"\\n  ... and {len(available_cells) - 20} more"
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
    
    # Limit to max_frames
    if len(frames) > max_frames:
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

'''

# 組合新內容
new_content = before_func + new_function + after_func

with open('analyze_trajectory_curvature.py', 'w') as f:
    f.write(new_content)

print("✓ 已完整替換函數")

# 驗證語法
import py_compile
try:
    py_compile.compile('analyze_trajectory_curvature.py', doraise=True)
    print("✓ 語法檢查通過")
except py_compile.PyCompileError as e:
    print(f"❌ 語法錯誤: {e}")
    print("恢復備份...")
    import shutil
    shutil.copy('analyze_trajectory_curvature.py.bak', 'analyze_trajectory_curvature.py')
    exit(1)
PYEOF

echo ""
echo "============================================================"
echo "修復完成！現在可以測試："
echo "============================================================"
echo ""
echo "cd /staging/groups/bhaskar_group/rho9/ivf_analysis"
echo "python3 scripts/analyze_trajectory_curvature.py --video_name ZS435-5"
echo ""

