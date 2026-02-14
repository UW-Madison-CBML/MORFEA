"""
Visualize TPHATE Segments with Original Frames
- Divide 3D TPHATE trajectory into 4 segments
- Create 4 colored blocks showing original frames for each segment
- Professional visualization for manifold learning papers
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
import argparse
import json
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from collections import defaultdict


def divide_trajectory_into_segments(Z_tphate, cell_id, frame_in_cell, n_segments=4):
    """
    将轨迹分成 n_segments 个段
    
    Args:
        Z_tphate: [N, 3] TPHATE embedding
        cell_id: [N] cell IDs
        frame_in_cell: [N] frame indices
        n_segments: 分段数量
    
    Returns:
        segment_labels: [N] segment labels (0, 1, 2, 3)
    """
    # 按 cell 分组，然后按时间排序
    unique_cells = np.unique(cell_id)
    segment_labels = np.zeros(len(Z_tphate), dtype=int)
    
    for cid in unique_cells:
        mask = cell_id == cid
        if mask.sum() == 0:
            continue
        
        # 获取该 cell 的轨迹点
        cell_indices = np.where(mask)[0]
        cell_frames = frame_in_cell[cell_indices]
        
        # 按时间排序
        sorted_order = np.argsort(cell_frames)
        sorted_indices = cell_indices[sorted_order]
        
        # 计算轨迹长度（arc length）
        cell_trajectory = Z_tphate[sorted_indices]
        if len(cell_trajectory) > 1:
            # 计算累积弧长
            diffs = np.diff(cell_trajectory, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            cumulative_length = np.concatenate([[0], np.cumsum(distances)])
            total_length = cumulative_length[-1]
            
            # 按累积弧长分成 n_segments 段
            segment_size = total_length / n_segments
            for i, idx in enumerate(sorted_indices):
                # 找到这个点在轨迹中的位置
                pos_in_sorted = np.where(sorted_indices == idx)[0][0]
                if pos_in_sorted == 0:
                    segment = 0
                else:
                    segment = min(int(cumulative_length[pos_in_sorted] / segment_size), n_segments - 1)
                segment_labels[idx] = segment
        else:
            segment_labels[sorted_indices[0]] = 0
    
    return segment_labels


def load_original_images(paths, index_csv=None, resize=128):
    """
    加载原始图片
    
    Args:
        paths: List of image paths (可能不存在，需要从 index.csv 重建)
        index_csv: index.csv 路径（用于重建正确的路径）
        resize: Target size
    
    Returns:
        images: List of PIL Images
    """
    images = []
    
    # 如果提供了 index_csv，尝试从那里加载路径
    if index_csv and Path(index_csv).exists():
        try:
            df = pd.read_csv(index_csv)
            print(f"  Loading images from index.csv...")
        except:
            df = None
    else:
        df = None
    
    for i, path_str in enumerate(paths):
        img = None
        
        # 方法 1: 尝试直接加载原始路径
        if path_str and Path(path_str).exists():
            try:
                img = Image.open(path_str)
            except:
                pass
        
        # 方法 2: 如果原始路径不存在，尝试从 index.csv 重建
        if img is None and df is not None:
            # 从 index.csv 中找到对应的路径
            # 需要根据 sequence_idx 和 frame_in_cell 来匹配
            # 这里简化处理：直接尝试所有可能的路径
            for idx, row in df.iterrows():
                row_paths = row['paths'].split('|')
                for p in row_paths:
                    # 尝试修复路径（如果是本地路径，替换为 CHTC 路径）
                    p_fixed = str(p)
                    # 替换常见的本地路径前缀
                    if '/Users/grnho/Desktop/Project IVF/embryo_dataset' in p_fixed:
                        p_fixed = p_fixed.replace('/Users/grnho/Desktop/Project IVF/embryo_dataset', 
                                                  '/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset')
                    elif '/Users/grnho/Desktop/Project IVF/' in p_fixed:
                        p_fixed = p_fixed.replace('/Users/grnho/Desktop/Project IVF/', 
                                                  '/staging/groups/bhaskar_group/rho9/ivf_data/')
                    
                    # 也尝试 data 符号链接
                    if not Path(p_fixed).exists():
                        # 尝试使用 data 符号链接
                        p_symlink = p_fixed.replace('/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset', 'data')
                        if Path(p_symlink).exists():
                            p_fixed = p_symlink
                    
                    if Path(p_fixed).exists():
                        try:
                            img = Image.open(p_fixed)
                            break
                        except:
                            pass
                if img is not None:
                    break
        
        # 方法 3: 如果还是找不到，创建灰色占位图
        if img is None:
            img = Image.new("L", (resize, resize), 128)  # 灰色而不是黑色
        else:
            img = img.convert("L")  # Grayscale
            img = img.resize((resize, resize), Image.BILINEAR)
        
        images.append(img)
    
    return images


def create_all_segments_frames_image(
    segment_labels, cell_id, frame_in_cell,
    segment_colors, segment_names,
    latents_file, index_csv, output_path,
    frames_per_row=20, frame_size=64
):
    """
    创建一张大图，显示每个 segment 对应的所有 frame 图像
    """
    
    # Load paths from latents file
    latents_data = np.load(latents_file, allow_pickle=True)
    paths = latents_data['paths'] if 'paths' in latents_data else None
    
    # Load index.csv if available
    df_index = None
    if Path(index_csv).exists():
        df_index = pd.read_csv(index_csv)
        print(f"  Loaded index.csv: {len(df_index)} sequences")
    
    n_segments = len(segment_names)
    
    # Calculate total canvas size
    max_frames_per_seg = max([(segment_labels == seg).sum() for seg in range(n_segments)])
    max_rows_per_seg = (max_frames_per_seg + frames_per_row - 1) // frames_per_row
    
    # Canvas: 4 segments stacked vertically
    segment_height = max_rows_per_seg * frame_size + 60  # 60 for title
    canvas_height = n_segments * segment_height + 40  # 40 for spacing
    canvas_width = frames_per_row * frame_size + 40
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load font
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
    
    y_offset = 20
    
    for seg in range(n_segments):
        mask = segment_labels == seg
        if mask.sum() == 0:
            continue
        
        seg_indices = np.where(mask)[0]
        seg_frames = frame_in_cell[mask]
        seg_cell_ids = cell_id[mask]
        
        # Sort by frame
        sorted_order = np.argsort(seg_frames)
        sorted_indices = seg_indices[sorted_order]
        
        n_frames = len(sorted_indices)
        n_rows = (n_frames + frames_per_row - 1) // frames_per_row
        
        # Draw segment title
        title = f"{segment_names[seg]} ({n_frames} frames)"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (canvas_width - title_width) // 2
        
        color_rgb = tuple(int(segment_colors[seg][i:i+2], 16) for i in (1, 3, 5))
        draw.rectangle(
            [title_x - 10, y_offset - 5, title_x + title_width + 10, y_offset + 35],
            fill=color_rgb,
            outline='black',
            width=2
        )
        draw.text((title_x, y_offset), title, fill='white', font=title_font)
        
        y_offset += 50
        
        # Load and paste all frames for this segment
        for i, idx in enumerate(sorted_indices):
            row = i // frames_per_row
            col = i % frames_per_row
            x_offset = col * frame_size + 20
            current_y = y_offset + row * frame_size
            
            current_frame_in_cell = seg_frames[sorted_order[i]]
            current_cell_id = seg_cell_ids[sorted_order[i]]
            
            img_rgb = None
            
            # Try to load image from paths
            if paths is not None and idx < len(paths) and paths[idx]:
                img_path = str(paths[idx])
                paths_to_try = [img_path]
                
                # Try various path formats
                if '/mnt/htc-cephfs/fuse/root' in img_path:
                    paths_to_try.append(img_path.replace('/mnt/htc-cephfs/fuse/root', ''))
                if 'embryo_dataset/' in img_path:
                    rel_path = img_path.split('embryo_dataset/')[-1]
                    paths_to_try.append(str(Path('data') / rel_path))
                
                for p in paths_to_try:
                    if Path(p).exists():
                        try:
                            img = Image.open(p)
                            img = img.convert("L")
                            img = img.resize((frame_size, frame_size), Image.BILINEAR)
                            img_rgb = img.convert('RGB')
                            break
                        except:
                            continue
            
            # Fallback to index.csv
            if img_rgb is None and df_index is not None:
                matching_rows = df_index[
                    (df_index['cell_id'] == current_cell_id) & 
                    (df_index['start_idx'] <= current_frame_in_cell) &
                    (df_index['start_idx'] + 16 > current_frame_in_cell)
                ]
                if len(matching_rows) > 0:
                    row_data = matching_rows.iloc[0]
                    paths_str = row_data['paths']
                    if pd.notna(paths_str):
                        path_list = paths_str.split('|')
                        t_in_seq = current_frame_in_cell - row_data['start_idx']
                        if 0 <= t_in_seq < len(path_list):
                            img_path = path_list[t_in_seq]
                            paths_to_try = [img_path]
                            if '/mnt/htc-cephfs/fuse/root' in img_path:
                                paths_to_try.append(img_path.replace('/mnt/htc-cephfs/fuse/root', ''))
                            if 'embryo_dataset/' in img_path:
                                rel_path = img_path.split('embryo_dataset/')[-1]
                                paths_to_try.append(str(Path('data') / rel_path))
                            
                            for p in paths_to_try:
                                if Path(p).exists():
                                    try:
                                        img = Image.open(p)
                                        img = img.convert("L")
                                        img = img.resize((frame_size, frame_size), Image.BILINEAR)
                                        img_rgb = img.convert('RGB')
                                        break
                                    except:
                                        continue
            
            # Fallback to gray placeholder
            if img_rgb is None:
                img_rgb = Image.new('RGB', (frame_size, frame_size), (128, 128, 128))
            
            canvas.paste(img_rgb, (x_offset, current_y))
            
            # Add frame number label
            label_text = f"F{current_frame_in_cell}"
            draw.text((x_offset + 2, current_y + 2), label_text, fill='yellow', font=label_font, stroke_width=1, stroke_fill='black')
        
        y_offset += n_rows * frame_size + 20
    
    # Save
    all_frames_file = output_path / "all_segments_all_frames.png"
    canvas.save(all_frames_file, 'PNG', dpi=(300, 300))
    print(f"✓ Saved comprehensive frames visualization to: {all_frames_file}")


def create_segment_visualization(
    tphate_file="tphate_3d_results.npz",
    latents_file="latents_all_frames.npz",
    index_csv="index.csv",
    output_dir="tphate_segments",
    n_segments=4,
    frames_per_row=10,
    frame_size=64
):
    """
    创建 4 个 segments 的对照图
    
    Args:
        tphate_file: TPHATE 结果文件
        latents_file: 原始 latents 文件（包含 paths）
        output_dir: 输出目录
        n_segments: 分段数量
        frames_per_row: 每行显示的 frames 数
        frame_size: 每个 frame 的显示大小
    """
    print("=== Creating TPHATE Segment Visualizations ===")
    
    # Load TPHATE data
    print(f"\nLoading TPHATE data from: {tphate_file}")
    tphate_data = np.load(tphate_file, allow_pickle=True)
    Z_tphate = tphate_data['Z_tphate']  # [N, 3]
    cell_id = tphate_data['cell_id']
    frame_in_cell = tphate_data['frame_in_cell']
    sequence_idx = tphate_data.get('sequence_idx', None)  # 从 tphate 文件获取，如果没有则从 latents 文件获取
    
    # Load original latents data (for paths)
    print(f"Loading original data from: {latents_file}")
    latents_data = np.load(latents_file, allow_pickle=True)
    paths = latents_data['paths'] if 'paths' in latents_data else None
    # 如果 tphate 文件没有 sequence_idx，从 latents 文件获取
    if sequence_idx is None:
        sequence_idx = latents_data['sequence_idx'] if 'sequence_idx' in latents_data else None
    
    # Check if index.csv exists for path reconstruction
    if Path(index_csv).exists():
        use_original_images = True
        print(f"  Found index.csv: {index_csv}")
        if paths is None:
            print("  ⚠️  No paths in latents file, will reconstruct from index.csv")
    elif paths is None:
        print("⚠️  Warning: No paths found and no index.csv. Cannot load original images.")
        print("   Will create visualization without original frames.")
        use_original_images = False
    else:
        use_original_images = True
    
    print(f"  Total points: {len(Z_tphate)}")
    print(f"  Unique cells: {len(np.unique(cell_id))}")
    print(f"  Cell IDs in TPHATE: {np.unique(cell_id)}")
    
    # Divide into segments
    print(f"\nDividing trajectory into {n_segments} segments...")
    segment_labels = divide_trajectory_into_segments(Z_tphate, cell_id, frame_in_cell, n_segments)
    
    # Count frames per segment
    segment_counts = {}
    for seg in range(n_segments):
        count = (segment_labels == seg).sum()
        segment_counts[seg] = count
        print(f"  Segment {seg}: {count} frames")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Color scheme for segments
    segment_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']  # Red, Teal, Blue, Orange
    segment_names = ['Segment A', 'Segment B', 'Segment C', 'Segment D']
    
    # 1. Create 3D plot with GRADIENT colors (frame-based, NOT normalized)
    print("\nCreating 3D TPHATE plot with gradient colors (frame-based, raw values)...")
    fig_3d_gradient = plt.figure(figsize=(14, 10))
    ax_3d_gradient = fig_3d_gradient.add_subplot(111, projection='3d')
    
    # Use frame_in_cell as color (raw values, NOT normalized)
    frame_values = frame_in_cell.astype(float)
    frame_min = frame_values.min()
    frame_max = frame_values.max()
    
    # Create colormap with explicit vmin/vmax to preserve raw frame values
    scatter = ax_3d_gradient.scatter(
        Z_tphate[:, 0],
        Z_tphate[:, 1],
        Z_tphate[:, 2],
        c=frame_values,  # Raw frame values
        cmap='viridis',  # or 'plasma', 'inferno', 'magma'
        vmin=frame_min,  # Explicitly set min to preserve raw values
        vmax=frame_max,  # Explicitly set max to preserve raw values
        s=30,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.3
    )
    
    # Plot trajectory lines (colored by frame)
    unique_cells = np.unique(cell_id)
    for cid in unique_cells:
        cell_mask = cell_id == cid
        if cell_mask.sum() > 1:
            indices = np.where(cell_mask)[0]
            frame_vals = frame_in_cell[indices]
            sorted_order = np.argsort(frame_vals)
            sorted_indices = indices[sorted_order]
            
            # Plot line with gradient colors (using raw frame values, normalized only for colormap)
            for i in range(len(sorted_indices) - 1):
                idx1, idx2 = sorted_indices[i], sorted_indices[i+1]
                # Normalize only for colormap lookup, but use raw value
                raw_frame_val = frame_values[idx1]
                color_val = (raw_frame_val - frame_min) / (frame_max - frame_min + 1e-6)
                ax_3d_gradient.plot(
                    [Z_tphate[idx1, 0], Z_tphate[idx2, 0]],
                    [Z_tphate[idx1, 1], Z_tphate[idx2, 1]],
                    [Z_tphate[idx1, 2], Z_tphate[idx2, 2]],
                    color=plt.cm.viridis(color_val),
                    alpha=0.4,
                    linewidth=1.5
                )
    
    ax_3d_gradient.set_xlabel('TPHATE Dimension 1', fontsize=12)
    ax_3d_gradient.set_ylabel('TPHATE Dimension 2', fontsize=12)
    ax_3d_gradient.set_zlabel('TPHATE Dimension 3', fontsize=12)
    ax_3d_gradient.set_title('3D TPHATE Trajectory (Gradient: Frame Index)', fontsize=14, fontweight='bold')
    
    # Add colorbar (showing raw frame values, NOT normalized)
    # 重要：使用 ScalarMappable 和 Normalize 来确保 colorbar 显示原始值（不归一化）
    print(f"  Frame values range: {frame_min} - {frame_max} (NOT normalized)")
    norm = Normalize(vmin=frame_min, vmax=frame_max)
    sm = ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])  # 空数组，只用于 colorbar
    cbar = plt.colorbar(sm, ax=ax_3d_gradient, pad=0.1)
    cbar.set_label('Frame Index (Raw, NOT Normalized)', fontsize=10)
    # 确保 colorbar 显示实际的原始 frame 值（不归一化）
    # 使用整数 ticks，显示实际的 frame 编号
    if frame_max - frame_min > 0:
        num_ticks = min(10, int(frame_max - frame_min + 1))
        if num_ticks > 1:
            ticks = np.linspace(frame_min, frame_max, num_ticks)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f'{int(x)}' for x in ticks])
        else:
            cbar.set_ticks([frame_min, frame_max])
            cbar.set_ticklabels([f'{int(frame_min)}', f'{int(frame_max)}'])
    else:
        cbar.set_ticks([frame_min])
        cbar.set_ticklabels([f'{int(frame_min)}'])
    print(f"  Colorbar range: {frame_min} - {frame_max} (raw values, NOT 0-1)")
    
    plt_3d_gradient_file = output_path / "tphate_3d_gradient.png"
    plt.tight_layout()
    plt.savefig(plt_3d_gradient_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved gradient 3D plot to: {plt_3d_gradient_file}")
    plt.close()
    
    # 2. Create 3D plot with colored segments (4 segments)
    print("\nCreating 3D TPHATE plot with colored segments...")
    fig_3d_segments = plt.figure(figsize=(14, 10))
    ax_3d_segments = fig_3d_segments.add_subplot(111, projection='3d')
    
    for seg in range(n_segments):
        mask = segment_labels == seg
        if mask.sum() == 0:
            continue
        
        # Plot points
        ax_3d_segments.scatter(
            Z_tphate[mask, 0],
            Z_tphate[mask, 1],
            Z_tphate[mask, 2],
            c=segment_colors[seg],
            label=segment_names[seg],
            s=30,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Plot trajectory lines for this segment
        unique_cells = np.unique(cell_id[mask])
        for cid in unique_cells:
            cell_mask = (cell_id == cid) & mask
            if cell_mask.sum() > 1:
                indices = np.where(cell_mask)[0]
                frame_values = frame_in_cell[indices]
                sorted_order = np.argsort(frame_values)
                sorted_indices = indices[sorted_order]
                
                ax_3d_segments.plot(
                    Z_tphate[sorted_indices, 0],
                    Z_tphate[sorted_indices, 1],
                    Z_tphate[sorted_indices, 2],
                    color=segment_colors[seg],
                    alpha=0.3,
                    linewidth=2
                )
    
    ax_3d_segments.set_xlabel('TPHATE Dimension 1', fontsize=12)
    ax_3d_segments.set_ylabel('TPHATE Dimension 2', fontsize=12)
    ax_3d_segments.set_zlabel('TPHATE Dimension 3', fontsize=12)
    ax_3d_segments.set_title('3D TPHATE Trajectory with 4 Segments', fontsize=14, fontweight='bold')
    ax_3d_segments.legend(loc='upper left', fontsize=10)
    
    plt_3d_segments_file = output_path / "tphate_3d_segments.png"
    plt.tight_layout()
    plt.savefig(plt_3d_segments_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved segments 3D plot to: {plt_3d_segments_file}")
    plt.close()
    
    # 2. Create 4 segment visualization blocks
    print("\nCreating segment visualization blocks...")
    
    for seg in range(n_segments):
        mask = segment_labels == seg
        if mask.sum() == 0:
            continue
        
        seg_indices = np.where(mask)[0]
        seg_frames = frame_in_cell[mask]
        seg_cell_ids = cell_id[mask]
        
        # Sort by frame
        sorted_order = np.argsort(seg_frames)
        sorted_indices = seg_indices[sorted_order]
        
        n_frames = len(sorted_indices)
        n_rows = (n_frames + frames_per_row - 1) // frames_per_row
        
        # Create canvas
        canvas_width = frames_per_row * (frame_size + 20) + 40
        canvas_height = n_rows * (frame_size + 40) + 100
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        
        # Load and paste frames
        if use_original_images:
            # 从 index.csv 加载正确的路径
            df_index = None
            if Path(index_csv).exists():
                try:
                    df_index = pd.read_csv(index_csv)
                    print(f"  Loading images from index.csv for {segment_names[seg]}...")
                    print(f"  TPHATE cell_id: {np.unique(seg_cell_ids)}")
                    print(f"  Index.csv has {len(df_index)} sequences")
                    print(f"  Index.csv unique cells: {df_index['cell_id'].unique()[:5]}...")  # 显示前5个
                except Exception as e:
                    print(f"  Error loading index.csv: {e}")
                    pass
            
            for i, idx in enumerate(sorted_indices):
                row = i // frames_per_row
                col = i % frames_per_row
                
                x_offset = 20 + col * (frame_size + 20)
                y_offset = 80 + row * (frame_size + 40)
                
                # Load image - 尝试多种方法
                img_rgb = None
                
                # 获取当前 frame 的信息
                current_frame_in_cell = seg_frames[sorted_order[i]]
                current_seq_idx = sequence_idx[idx] if sequence_idx is not None and idx < len(sequence_idx) else None
                
                # 方法 1: 从 paths 数组（最直接的方法，适用于直接从 cell folder 提取的情况）
                if paths is not None and idx < len(paths) and paths[idx]:
                    img_path = str(paths[idx])
                    
                    # 路径修复策略（按优先级）
                    paths_to_try = [img_path]
                    
                    # 策略1: 去掉 /mnt/htc-cephfs/fuse/root/ 前缀
                    if '/mnt/htc-cephfs/fuse/root' in img_path:
                        paths_to_try.append(img_path.replace('/mnt/htc-cephfs/fuse/root', ''))
                    
                    # 策略2: 使用 data 符号链接
                    if 'embryo_dataset/' in img_path:
                        rel_path = img_path.split('embryo_dataset/')[-1]
                        paths_to_try.append(str(Path('data') / rel_path))
                    
                    # 策略3: 尝试本地路径替换（fallback）
                    if '/Users/grnho/Desktop/Project IVF/embryo_dataset' in img_path:
                        paths_to_try.append(img_path.replace('/Users/grnho/Desktop/Project IVF/embryo_dataset',
                                                  '/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset'))
                    
                    # 尝试所有路径
                    for p in paths_to_try:
                        if Path(p).exists():
                            try:
                                img = Image.open(p)
                                img = img.convert("L")
                                img = img.resize((frame_size, frame_size), Image.BILINEAR)
                                img_rgb = img.convert('RGB')
                                break
                            except Exception as e:
                                if i < 3:  # 只打印前3个错误
                                    print(f"    [Debug] Error loading {p[:60]}...: {e}")
                                continue
                
                # 方法 2: 从 index.csv 重建路径（最可靠的方法）
                if img_rgb is None and df_index is not None and current_seq_idx is not None:
                    if current_seq_idx < len(df_index):
                        row = df_index.iloc[current_seq_idx]
                        
                        # 验证 cell_id 匹配
                        row_cell_id = row['cell_id']
                        current_cell_id = seg_cell_ids[sorted_order[i]]
                        
                        if row_cell_id != current_cell_id:
                            # Cell ID 不匹配，尝试找到正确的序列
                            # 在 index.csv 中查找匹配的 cell_id 和 frame_in_cell
                            matching_rows = df_index[
                                (df_index['cell_id'] == current_cell_id) & 
                                (df_index['start_idx'] <= current_frame_in_cell) &
                                (df_index['start_idx'] + 16 > current_frame_in_cell)  # 假设序列长度是16
                            ]
                            if len(matching_rows) > 0:
                                # 使用第一个匹配的行
                                row = matching_rows.iloc[0]
                                current_seq_idx = matching_rows.index[0]
                        
                        row_paths = row['paths'].split('|')
                        start_idx = int(row['start_idx'])
                        
                        # frame_in_cell 是全局索引，需要转换为序列内的相对位置
                        # frame_in_cell = start_idx + t，所以 t = frame_in_cell - start_idx
                        t_in_sequence = current_frame_in_cell - start_idx
                        
                        # 检查 t_in_sequence 是否在有效范围内
                        if 0 <= t_in_sequence < len(row_paths):
                            img_path = row_paths[t_in_sequence]
                            
                            # 路径已经在 index.csv 中是正确的，直接使用
                            # 但需要检查是否存在，如果不存在，尝试其他路径格式
                            
                            # 如果路径已经存在，直接使用
                            if not Path(img_path).exists():
                                # 尝试不同的路径格式
                                # 格式1: 去掉 /mnt/htc-cephfs/fuse/root/ 前缀（如果存在）
                                if '/mnt/htc-cephfs/fuse/root' in img_path:
                                    img_path_alt = img_path.replace('/mnt/htc-cephfs/fuse/root', '')
                                    if Path(img_path_alt).exists():
                                        img_path = img_path_alt
                                
                                # 格式2: 使用 data 符号链接
                                if not Path(img_path).exists():
                                    # 提取相对路径部分
                                    if 'embryo_dataset/' in img_path:
                                        rel_path = img_path.split('embryo_dataset/')[-1]
                                        img_path_symlink = Path('data') / rel_path
                                        if img_path_symlink.exists():
                                            img_path = str(img_path_symlink)
                                
                                # 格式3: 尝试本地路径替换（fallback）
                                if not Path(img_path).exists():
                                    if '/Users/grnho/Desktop/Project IVF/embryo_dataset' in img_path:
                                        img_path = img_path.replace('/Users/grnho/Desktop/Project IVF/embryo_dataset',
                                                          '/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset')
                                    elif '/Users/grnho/Desktop/Project IVF/' in img_path:
                                        img_path = img_path.replace('/Users/grnho/Desktop/Project IVF/',
                                                          '/staging/groups/bhaskar_group/rho9/ivf_data/')
                            
                            if Path(img_path).exists():
                                try:
                                    img = Image.open(img_path)
                                    img = img.convert("L")
                                    img = img.resize((frame_size, frame_size), Image.BILINEAR)
                                    img_rgb = img.convert('RGB')
                                except Exception as e:
                                    # 调试信息
                                    if i < 5:  # 只打印前5个错误
                                        print(f"    [Debug] Error loading image {img_path[:60]}...: {e}")
                                    pass
                
                # 方法 3: 如果还是找不到，使用灰色占位图
                if img_rgb is None:
                    img_rgb = Image.new('RGB', (frame_size, frame_size), (128, 128, 128))
                
                # Paste image
                canvas.paste(img_rgb, (x_offset, y_offset))
                
                # Add frame number label
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(canvas)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
                except:
                    font = ImageFont.load_default()
                
                # 使用 current_frame_in_cell（已经在上面定义了）
                text = f"Frame {current_frame_in_cell}"
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_x = x_offset + (frame_size - text_width) // 2
                text_y = y_offset + frame_size + 5
                
                # Draw text with background
                draw.rectangle(
                    [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + 14],
                    fill='white',
                    outline='black',
                    width=1
                )
                draw.text((text_x, text_y), text, fill='black', font=font)
        
        # Add title
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            title_font = ImageFont.load_default()
        
        title = f"{segment_names[seg]} ({n_frames} frames)"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (canvas_width - title_width) // 2
        title_y = 20
        
        # Draw title with colored background
        color_rgb = tuple(int(segment_colors[seg][i:i+2], 16) for i in (1, 3, 5))
        draw.rectangle(
            [title_x - 10, title_y - 5, title_x + title_width + 10, title_y + 30],
            fill=color_rgb,
            outline='black',
            width=2
        )
        draw.text((title_x, title_y), title, fill='white', font=title_font)
        
        # Save segment visualization
        seg_file = output_path / f"segment_{chr(65+seg)}_frames.png"  # A, B, C, D
        canvas.save(seg_file, 'PNG', dpi=(300, 300))
        print(f"✓ Saved {segment_names[seg]} to: {seg_file}")
    
    # 3. Create combined visualization (all 4 segments in one image)
    print("\nCreating combined visualization...")
    fig_combined = plt.figure(figsize=(20, 20))
    
    for seg in range(n_segments):
        mask = segment_labels == seg
        
        # Create 3D subplot for this segment
        ax_3d_seg = fig_combined.add_subplot(2, 2, seg + 1, projection='3d')
        
        if mask.sum() == 0:
            ax_3d_seg.text(0, 0, 0, f"{segment_names[seg]}\n(No frames)", 
                   ha='center', va='center', fontsize=16)
            ax_3d_seg.set_title(f"{segment_names[seg]} (No frames)", fontsize=14)
            continue
        
        # Plot all points (gray background)
        ax_3d_seg.scatter(
            Z_tphate[:, 0],
            Z_tphate[:, 1],
            Z_tphate[:, 2],
            c='lightgray',
            s=10,
            alpha=0.3
        )
        
        # Highlight this segment
        ax_3d_seg.scatter(
            Z_tphate[mask, 0],
            Z_tphate[mask, 1],
            Z_tphate[mask, 2],
            c=segment_colors[seg],
            s=50,
            alpha=0.8,
            edgecolors='black',
            linewidths=1,
            label=segment_names[seg]
        )
        
        # Plot trajectory lines
        unique_cells = np.unique(cell_id[mask])
        for cid in unique_cells:
            cell_mask = (cell_id == cid) & mask
            if cell_mask.sum() > 1:
                indices = np.where(cell_mask)[0]
                frame_values = frame_in_cell[indices]
                sorted_order = np.argsort(frame_values)
                sorted_indices = indices[sorted_order]
                
                ax_3d_seg.plot(
                    Z_tphate[sorted_indices, 0],
                    Z_tphate[sorted_indices, 1],
                    Z_tphate[sorted_indices, 2],
                    color=segment_colors[seg],
                    alpha=0.5,
                    linewidth=2
                )
        
        ax_3d_seg.set_title(f"{segment_names[seg]} ({mask.sum()} frames)", 
                           fontsize=14, fontweight='bold', color=segment_colors[seg])
        ax_3d_seg.set_xlabel('Dim 1', fontsize=10)
        ax_3d_seg.set_ylabel('Dim 2', fontsize=10)
        ax_3d_seg.set_zlabel('Dim 3', fontsize=10)
    
    plt.suptitle('3D TPHATE Trajectory Segments', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    combined_file = output_path / "tphate_segments_combined.png"
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined visualization to: {combined_file}")
    plt.close()
    
    # 4. Create a single large image showing ALL frames for each segment
    print("\nCreating comprehensive segment frames visualization (all frames for each segment)...")
    try:
        create_all_segments_frames_image(
            segment_labels, cell_id, frame_in_cell,
            segment_colors, segment_names,
            latents_file, index_csv, output_path,
            frames_per_row=20, frame_size=64
        )
    except Exception as e:
        print(f"  ⚠️  Error creating all segments frames image: {e}")
        import traceback
        traceback.print_exc()
    
    # Save segment metadata
    metadata = {
        "n_segments": n_segments,
        "segment_counts": {f"Segment {chr(65+i)}": int(count) for i, count in segment_counts.items()},
        "segment_colors": segment_colors,
        "total_frames": int(len(Z_tphate))
    }
    
    metadata_file = output_path / "segments_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved metadata to: {metadata_file}")
    print(f"\n✅ All visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - tphate_3d_gradient.png (3D plot with gradient colors showing frame progression)")
    print("  - tphate_3d_segments.png (3D plot with colored segments)")
    print("  - segment_A_frames.png (Segment A original frames)")
    print("  - segment_B_frames.png (Segment B original frames)")
    print("  - segment_C_frames.png (Segment C original frames)")
    print("  - segment_D_frames.png (Segment D original frames)")
    print("  - tphate_segments_combined.png (Combined 4-segment view)")
    print("  - all_segments_all_frames.png (ALL frames for each segment in one image)")
    
    return segment_labels, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create TPHATE segment visualizations")
    parser.add_argument("--tphate_file", type=str, default="tphate_3d_results.npz",
                       help="TPHATE results .npz file")
    parser.add_argument("--latents_file", type=str, default="latents_all_frames.npz",
                       help="Original latents .npz file (with paths)")
    parser.add_argument("--index_csv", type=str, default="index.csv",
                       help="Index CSV file (for reconstructing image paths)")
    parser.add_argument("--output_dir", type=str, default="tphate_segments",
                       help="Output directory")
    parser.add_argument("--n_segments", type=int, default=4,
                       help="Number of segments")
    parser.add_argument("--frames_per_row", type=int, default=10,
                       help="Number of frames per row in segment visualization")
    parser.add_argument("--frame_size", type=int, default=64,
                       help="Size of each frame in visualization")
    
    args = parser.parse_args()
    
    create_segment_visualization(
        tphate_file=args.tphate_file,
        latents_file=args.latents_file,
        index_csv=args.index_csv if args.index_csv else "index.csv",
        output_dir=args.output_dir,
        n_segments=args.n_segments,
        frames_per_row=args.frames_per_row,
        frame_size=args.frame_size
    )

