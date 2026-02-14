# Trajectory Curvature Analysis

This script analyzes embryo trajectory curvature using a circle-fitting method on 3D TPHATE embeddings.

## Features

1. **Loads latent vectors** for a given video_name (cell ID)
2. **Computes 3D TPHATE embedding** from latent vectors
3. **Calculates curvature** along the trajectory using circle-fitting on triplets of consecutive points
4. **Visualizes trajectory** colored by curvature
5. **Identifies high-curvature regions** (default: 95th percentile)
6. **Extracts and saves** high-curvature frames

## Usage

### Basic usage:

```bash
python scripts/analyze_trajectory_curvature.py --video_name ZS435-5
```

### With custom options:

```bash
python scripts/analyze_trajectory_curvature.py \
    --video_name ZS435-5 \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --data_root data \
    --output_dir curvature_analysis \
    --curvature_threshold_percentile 95.0 \
    --knn 5 \
    --max_frames 435
```

## Arguments

- `--video_name` (required): Cell ID, e.g., "ZS435-5" or "RS363-7"
- `--checkpoint`: Path to model checkpoint (default: `checkpoints/checkpoint_epoch_50.pt`)
- `--data_root`: Root directory containing cell folders (default: `data`)
- `--output_dir`: Output directory for results (default: `curvature_analysis`)
- `--device`: Device to run model on (default: `cuda` if available, else `cpu`)
- `--max_frames`: Maximum number of frames to process (default: 435)
- `--curvature_threshold_percentile`: Percentile threshold for high curvature (default: 95.0)
- `--knn`: Number of nearest neighbors for TPHATE (default: 5)

## Output

The script creates the following outputs in the specified `output_dir`:

```
curvature_analysis/
├── figures/
│   ├── tphate_curvature_{video_name}.png          # 3D trajectory colored by curvature
│   └── high_curvature_montage_{video_name}.png    # Grid of high-curvature frames
├── frames/
│   └── high_curvature/
│       ├── {video_name}_t{idx}.png                # Individual high-curvature frames
│       └── ...
└── curvature_data_{video_name}.npz                # Saved trajectory, curvatures, indices
```

## Curvature Calculation

The curvature is computed using a circle-fitting method:

1. For each point `p` at time `t`, consider the triplet `(p_prev, p, p_next)`
2. Compute side lengths: `a = ||p - p_prev||`, `b = ||p_next - p||`, `c = ||p_next - p_prev||`
3. Use Heron's formula for triangle area: `area = sqrt(s*(s-a)*(s-b)*(s-c))` where `s = (a+b+c)/2`
4. Compute curvature: `kappa = 4 * area / (a * b * c)`

High-curvature regions indicate sharp bends in the developmental trajectory, which may correspond to important developmental transitions.

## Example: Analyzing Good vs Poor Quality Embryos

```bash
# Good quality embryo (TE=A, ICM=A)
python scripts/analyze_trajectory_curvature.py --video_name ZS435-5

# Poor quality embryo (TE=C, ICM=B)
python scripts/analyze_trajectory_curvature.py --video_name RS363-7
```

## Requirements

- PyTorch
- NumPy
- Matplotlib
- Pillow (PIL)
- tphate (or phate as fallback)
- Trained autoencoder model checkpoint

## Notes

- The script automatically detects the model structure (Encoder+Decoder vs ConvLSTMAutoencoder)
- If TPHATE is not available, it falls back to PHATE
- The script processes frames directly from cell folders, not from index.csv
- Empty wells are automatically detected and skipped

