# Running Curvature Analysis on CHTC

## Quick Start

1. **Upload the script to CHTC:**
```bash
scp scripts/analyze_trajectory_curvature.py rho9@ap2001.chtc.wisc.edu:~/ivf_repo/
```

2. **SSH to CHTC:**
```bash
ssh rho9@ap2001.chtc.wisc.edu
cd ~/ivf_repo
```

3. **Run the analysis:**
```bash
# Good quality embryo
python3 scripts/analyze_trajectory_curvature.py --video_name ZS435-5

# Poor quality embryo
python3 scripts/analyze_trajectory_curvature.py --video_name RS363-7
```

## Full Command with Options

```bash
python3 scripts/analyze_trajectory_curvature.py \
    --video_name ZS435-5 \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --data_root data \
    --output_dir curvature_analysis \
    --device cuda \
    --max_frames 435 \
    --curvature_threshold_percentile 95.0 \
    --knn 5
```

## Expected Output

After running, you'll see:
```
============================================================
Trajectory Curvature Analysis
============================================================
Video: ZS435-5
Device: cuda
Output: curvature_analysis
============================================================

Loading model from checkpoints/checkpoint_epoch_50.pt...
✓ Loaded model (ConvLSTMAutoencoder structure)

Loading latent vectors for ZS435-5...
  Found cell directory: data/ZS435-5
  Processing 435 frames...
  ✓ Extracted 435 latent vectors, shape: (435, 256)

Computing 3D TPHATE embedding...
  Input shape: (435, 256)
  ✓ TPHATE embedding shape: (435, 3)

Computing curvature...
  ✓ Computed curvature for 435 points
    Min: 0.000000, Max: 0.123456, Mean: 0.012345

Plotting trajectory colored by curvature...
  ✓ Saved plot to curvature_analysis/figures/tphate_curvature_ZS435-5.png

High-curvature analysis:
  Threshold (95.0th percentile): 0.045678
  High-curvature timepoints: 22
  Indices: [45 67 89 ...]

Extracting 22 high-curvature frames...
  ✓ Saved 22 frames to curvature_analysis/frames/high_curvature

Creating curvature montage...
  ✓ Saved montage to curvature_analysis/figures/high_curvature_montage_ZS435-5.png

✓ Saved curvature data to curvature_analysis/curvature_data_ZS435-5.npz

============================================================
Analysis complete!
============================================================
Results saved to: curvature_analysis
  - Trajectory plot: curvature_analysis/figures/tphate_curvature_ZS435-5.png
  - Montage: curvature_analysis/figures/high_curvature_montage_ZS435-5.png
  - High-curvature frames: curvature_analysis/frames/high_curvature
  - Curvature data: curvature_analysis/curvature_data_ZS435-5.npz
```

## Troubleshooting

### If you get "command not found: python"
Use `python3` instead:
```bash
python3 scripts/analyze_trajectory_curvature.py --video_name ZS435-5
```

### If you get "Data root not found"
Make sure the `data` symlink exists:
```bash
cd ~/ivf_repo
ls -la data
# Should point to: /staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset

# If not, create it:
ln -s /staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset data
```

### If you get "Model checkpoint not found"
Check if checkpoint exists:
```bash
ls -lh checkpoints/checkpoint_epoch_50.pt
```

### If you get "tphate not available"
Install it:
```bash
pip install --user tphate
```

## Downloading Results

After analysis completes, download results to local machine:

```bash
# From your local machine
scp -r rho9@ap2001.chtc.wisc.edu:~/ivf_repo/curvature_analysis ./
```

Or download specific files:
```bash
scp rho9@ap2001.chtc.wisc.edu:~/ivf_repo/curvature_analysis/figures/*.png ./
```

