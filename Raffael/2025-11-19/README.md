# ConvLSTM Autoencoder - Complete High-Quality Version

## Overview

This is a complete, high-quality ConvLSTM Autoencoder implementation for IVF time-lapse video analysis.



## File Structure

```
Autoencoder_Decoder_ver02/
├── conv_lstm.py          # ConvLSTM implementation
├── model.py              # Complete model (Encoder + Decoder + Classifier)
├── losses.py             # Loss functions (MS-SSIM, L1, temporal smoothness)
├── train.py              # Complete training script
├── test_model.py         # Test script (ensures model can run)
├── verify_data_connection.py  # Data connection verification
└── README.md             # This file
```

## Model Architecture

### Input/Output

- **Input**: `(B, T, 1, 128, 128)` - video sequence
- **Output**:
  - `reconstruction`: `(B, T, 1, 128, 128)` - reconstructed video
  - `z_seq`: `(B, T, 256, 16, 16)` - full latent sequence
  - `z_last`: `(B, 256, 16, 16)` - last timestep latent
  - `logits`: `(B, 2)` - classification logits (if enabled)

### Encoder

1. **Spatial Compression** (2D CNN):
   - 128×128 → 64×64 → 32×32 → 16×16
   - Channels: 1 → 64 → 128 → 256

2. **Temporal Modeling** (ConvLSTM):
   - Input: `(B, T, 256, 16, 16)`
   - Output: `(B, T, 256, 16, 16)` (z_seq)
   - Last frame: `(B, 256, 16, 16)` (z_last)

### Decoder

1. **Temporal Decoding** (ConvLSTM):
   - Input: `(B, T, 256, 16, 16)`
   - Output: `(B, T, 128, 16, 16)`

2. **Spatial Reconstruction** (ConvTranspose):
   - 16×16 → 32×32 → 64×64 → 128×128
   - Channels: 128 → 64 → 32 → 1

### Classifier (Optional)

- Empty/Non-empty classification based on `z_last`
- Architecture: Global Average Pooling → FC(512) → FC(256) → FC(2)

## Loss Functions

### Reconstruction Loss

Combined loss: `0.5 * L1 + 0.5 * MS-SSIM`

- **L1 Loss**: Pixel-level reconstruction error
- **MS-SSIM Loss**: Multi-scale structural similarity (perceptual quality)

### Temporal Smoothness Loss

Encourages similar latents for adjacent timesteps: `0.1 * ||z_t - z_{t-1}||^2`

### Classification Loss (Optional)

If classifier enabled: `CrossEntropyLoss(logits, labels)`

## Usage

### 1. Test Model (Ensure It Can Run)

```bash
cd Code/Autoencoder_Decoder_ver02
python3 test_model.py
```

This verifies:
- Model can be created correctly
- Forward pass works
- Loss computation is correct
- Backward pass can execute
- Different batch sizes work

### 2. Verify Data Connection

```bash
python3 verify_data_connection.py
```

This verifies:
- Dataset loading works
- DataLoader batching is correct
- Model input/output shapes match
- Loss computation works
- Complete training step works

### 3. Train Model

```bash
python3 train.py \
    --index_csv ../index.csv \
    --batch_size 8 \
    --seq_len 20 \
    --num_epochs 50 \
    --learning_rate 3e-4 \
    --save_dir checkpoints \
    --log_dir logs
```

### 4. Train on CHTC H200

1. **Upload to GitHub**:
   ```bash
   git add Code/Autoencoder_Decoder_ver02/
   git commit -m "Add complete ConvLSTM Autoencoder"
   git push
   ```

2. **Setup on CHTC**:
   ```bash
   ssh rho9@ap2001.chtc.wisc.edu
   cd ~/ivf-embryo-analysis-Raffael
   git pull
   ```

3. **Modify training script to use new model**:
   - Update `train_ae.py` or create new training script
   - Use `Autoencoder_Decoder_ver02/train.py`

4. **Submit to CHTC**:
   ```bash
   condor_submit train_h200_lab.sub
   ```

## Configuration Parameters

### Model Parameters (High-Quality Configuration)

- `encoder_hidden_dim`: 256
- `encoder_layers`: 2
- `decoder_hidden_dim`: 128
- `decoder_layers`: 2
- `seq_len`: 20

### Training Parameters

- `batch_size`: 8
- `learning_rate`: 3e-4
- `weight_decay`: 1e-5
- `num_epochs`: 50
- `l1_weight`: 0.5
- `ms_ssim_weight`: 0.5
- `smooth_weight`: 0.1

## Output

### Checkpoints

Saved every 5 epochs, includes:
- Model weights
- Optimizer state
- Learning rate scheduler state
- Training configuration

### Training Logs

`logs/training_log.json` contains for each epoch:
- Total loss
- Reconstruction loss (L1 + MS-SSIM)
- Temporal smoothness loss
- Learning rate

## Post-Training Analysis

After training, you can use latents for:

1. **T-PHATE Analysis**:
   ```python
   z_last_flat = z_last.view(B, -1)  # (B, C*H*W)
   # Input to T-PHATE
   ```

2. **TDA Analysis**:
   ```python
   # Use z_seq or z_last for topological data analysis
   ```

3. **Classification Evaluation**:
   ```python
   # If labels available, evaluate classifier performance
   ```

## Important Notes

1. **GPU Memory**: This configuration uses significant GPU memory, ensure H200 has sufficient resources
2. **Training Time**: High-quality configuration requires longer training time
3. **Data Path**: Ensure `index.csv` path is correct
4. **Dependencies**: Requires `pytorch`, `opencv-python`, `pandas`, `numpy`, `tqdm`

## Compatibility with Existing Code

- Uses same `dataset_ivf.py`
- Input format: `(B, T, 1, 128, 128)`
- Output format: Dictionary containing `reconstruction`, `z_seq`, `z_last`, `logits`

## Latent Output for T-PHATE / TDA

The model exposes `z_seq` and `z_last` in the forward output for downstream analysis:

```python
output = model(x)
z_seq = output["z_seq"]   # (B, T, 256, 16, 16) - full temporal sequence
z_last = output["z_last"] # (B, 256, 16, 16) - last timestep latent

# For T-PHATE analysis
z_last_flat = z_last.view(B, -1)  # (B, 65536)
# Input to T-PHATE

# For TDA analysis
# Use z_seq or z_last for topological data analysis
```

## Data Connection

The model is **fully connected** to the data pipeline.

Data flow:
```
Dataset → DataLoader → Model → Loss → Backward
[T,1,128,128] → (B,T,1,128,128) → Model → Loss ✓
```

## Update History

- 2025-11-16: Initial version, complete high-quality implementation
