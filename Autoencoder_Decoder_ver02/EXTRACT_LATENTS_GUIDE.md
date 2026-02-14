# 📦 Extract All Latent Trajectories Guide

## 🎯 Overview

This script extracts latent trajectories (z_seq) from all embryos using a trained model and saves them in an organized structure.

## 📂 Output Structure

```
model_latents/
  model_version_name/
    checkpoint.pt              # Model checkpoint (copied)
    latents/
      embryo_ZS435-5.npy       # Latent trajectory [T, latent_dim]
      embryo_RS363-7.npy
      embryo_XXX-XX.npy
      ...
    metadata.json               # Extraction metadata
```

## 🚀 Usage

### Local Usage

```bash
python extract_all_latent_trajectories.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --model_version v1_baseline \
    --index_csv index.csv \
    --device cuda \
    --output_dir model_latents
```

### CHTC Usage

1. **Prepare submit file** (`extract_latents.sub`):
   ```bash
   # Edit extract_latents.sub to set:
   checkpoint = checkpoints/checkpoint_epoch_50.pt
   model_version = v1_baseline
   queue
   ```

2. **Submit job**:
   ```bash
   condor_submit extract_latents.sub
   ```

3. **Check status**:
   ```bash
   condor_q
   ```

## 📋 Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--checkpoint` | Yes | - | Path to model checkpoint |
| `--model_version` | Yes | - | Model version name (e.g., "v1_baseline") |
| `--index_csv` | No | "index.csv" | Path to index CSV file |
| `--data_root` | No | None | Root directory for data |
| `--device` | No | "cpu" | Device: "cpu" or "cuda" |
| `--batch_size` | No | 1 | Batch size for processing |
| `--max_embryos` | No | None | Max embryos to process (None = all) |
| `--output_dir` | No | "model_latents" | Base output directory |

## 📊 Output Files

### Latent Trajectory Files (`latents/embryo_*.npy`)

- **Format**: NumPy array
- **Shape**: `[T, latent_dim]`
  - `T`: Number of time steps (frames)
  - `latent_dim`: Latent dimension (usually 256 for encoder or 128 for decoder)
- **Content**: Latent vectors z_seq from model.encode()

### Metadata (`metadata.json`)

```json
{
  "model_version": "v1_baseline",
  "checkpoint_path": "checkpoints/checkpoint_epoch_50.pt",
  "extraction_date": "2024-01-01T12:00:00",
  "device": "cuda",
  "total_embryos": 50,
  "successful": 48,
  "failed": 2,
  "embryos": {
    "ZS435-5": {
      "status": "success",
      "num_sequences": 3,
      "trajectory_shape": [435, 256],
      "trajectory_dtype": "float32",
      "file": "latents/embryo_ZS435-5.npy"
    },
    ...
  }
}
```

## 🔍 Loading Saved Trajectories

```python
import numpy as np
from pathlib import Path

# Load a single trajectory
trajectory = np.load("model_latents/v1_baseline/latents/embryo_ZS435-5.npy")
print(f"Shape: {trajectory.shape}")  # [T, latent_dim]

# Load metadata
import json
with open("model_latents/v1_baseline/metadata.json") as f:
    metadata = json.load(f)

# List all embryos
embryos = list(metadata["embryos"].keys())
print(f"Total embryos: {len(embryos)}")
```

## 💡 Tips

1. **Model Version Naming**: Use descriptive names like:
   - `v1_baseline`
   - `v2_no_temporal_smooth`
   - `v3_different_loss_weights`
   - `v4_ablation_xyz`

2. **Testing**: Use `--max_embryos 5` to test on a small subset first

3. **Memory**: Large datasets may require more memory. Adjust `request_memory` in `.sub` file

4. **GPU**: For faster processing, use `--device cuda` and request GPU in CHTC

5. **Multiple Models**: Run the script multiple times with different `--model_version` to compare models

## 🔄 Integration with Experiment Data Manager

You can integrate this with the `ExperimentDataManager`:

```python
from experiment_data_manager import ExperimentDataManager
import numpy as np

# Load trajectories
trajectory = np.load("model_latents/v1_baseline/latents/embryo_ZS435-5.npy")

# Save to experiment manager
manager = ExperimentDataManager(base_dir="experiments")
manager.create_model_version("v1_baseline", model_config={...})
manager.save_data("v1_baseline", "ZS435-5", "latents", trajectory)
```

## ⚠️ Troubleshooting

### Checkpoint not found
- Make sure checkpoint path is correct
- On CHTC, ensure checkpoint is in `transfer_input_files`

### No sequences found for embryo
- Check that `index.csv` is correct
- Verify embryo ID format matches dataset

### Out of memory
- Reduce batch size
- Process fewer embryos at once
- Use CPU instead of GPU if GPU memory is limited

### Model architecture mismatch
- The script tries to auto-detect model parameters
- If it fails, you may need to manually specify model architecture in the code

## 📝 Example Workflow

```bash
# 1. Extract trajectories for baseline model
python extract_all_latent_trajectories.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --model_version v1_baseline \
    --device cuda

# 2. Extract trajectories for ablation model
python extract_all_latent_trajectories.py \
    --checkpoint checkpoints/checkpoint_epoch_50_ablation.pt \
    --model_version v2_no_smooth \
    --device cuda

# 3. Compare trajectories
python compare_trajectories.py \
    --model1 model_latents/v1_baseline \
    --model2 model_latents/v2_no_smooth
```

