# Running Curvature Analysis in Group Directory

## Setup in Group Directory

1. **Create scripts directory in group space:**
```bash
# On CHTC
mkdir -p /staging/groups/bhaskar_group/rho9/ivf_analysis/scripts
```

2. **Upload the script to group directory:**
```bash
# From your local machine
scp scripts/analyze_trajectory_curvature.py rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/ivf_analysis/scripts/
```

3. **SSH to CHTC and navigate to group directory:**
```bash
ssh rho9@ap2001.chtc.wisc.edu
cd /staging/groups/bhaskar_group/rho9/ivf_analysis
```

4. **Run the analysis (outputs will go to group directory):**
```bash
# Good quality embryo
python3 scripts/analyze_trajectory_curvature.py --video_name ZS435-5

# Poor quality embryo  
python3 scripts/analyze_trajectory_curvature.py --video_name RS363-7
```

## Output Location

By default, results will be saved to:
```
/staging/groups/bhaskar_group/rho9/curvature_analysis/
├── figures/
│   ├── tphate_curvature_ZS435-5.png
│   └── high_curvature_montage_ZS435-5.png
├── frames/
│   └── high_curvature/
│       └── ...
└── curvature_data_ZS435-5.npz
```

## Custom Output Directory

You can also specify a custom output directory in group space:

```bash
python3 scripts/analyze_trajectory_curvature.py \
    --video_name ZS435-5 \
    --output_dir /staging/groups/bhaskar_group/rho9/curvature_results
```

## Full Example with All Options

```bash
cd /staging/groups/bhaskar_group/rho9/ivf_analysis

python3 scripts/analyze_trajectory_curvature.py \
    --video_name ZS435-5 \
    --checkpoint ~/ivf_repo/checkpoints/checkpoint_epoch_50.pt \
    --data_root /staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset \
    --output_dir /staging/groups/bhaskar_group/rho9/curvature_analysis \
    --device cuda \
    --max_frames 435 \
    --curvature_threshold_percentile 95.0 \
    --knn 5
```

## Advantages of Using Group Directory

1. **Shared access**: Other group members can access results
2. **More storage**: Group directories typically have more quota
3. **Persistent**: Results stay in group space, not in home directory
4. **Organized**: Keeps analysis results separate from code

## Directory Structure Recommendation

```
/staging/groups/bhaskar_group/rho9/
├── ivf_data/                    # Original dataset
│   └── embryo_dataset/
├── ivf_analysis/                # Analysis scripts and code
│   └── scripts/
│       └── analyze_trajectory_curvature.py
└── curvature_analysis/           # Analysis results (auto-created)
    ├── figures/
    ├── frames/
    └── *.npz
```

## Downloading Results from Group Directory

```bash
# From your local machine
scp -r rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/curvature_analysis ./
```

## Notes

- Make sure you have write permissions in the group directory
- The script will automatically use group directory if available
- If group directory doesn't exist, it falls back to local `curvature_analysis/`

