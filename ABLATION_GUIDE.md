# Ablation Study Guide

This guide explains how to conduct ablation studies with the ConvLSTM Autoencoder training pipeline. All ablation settings are logged to WandB and saved to configuration files for reproducibility.

## Quick Reference

### Shell Script Syntax
```bash
./train_model.sh <MODE> <LOSS_TYPE> [OPTIONS]

# Example:
./train_model.sh convlstm_latent_split l1 --ms-ssim-weight 0.0 --no-convlstm
```

### Available Options

**Loss Ablations:**
- `--ms-ssim-weight <float>` - MS-SSIM loss weight (default: 0.5, disable: 0.0)
- `--rec-weight <float>` - Reconstruction loss weight (default: 0.5, disable: 0.0)
- `--temporal-weight <float>` - Temporal smoothness weight (default: 0.1, disable: 0.0)

**Model Ablations:**
- `--dropout-rate <float>` - Dropout rate (default: 0.1, disable: 0.0)
- `--no-convlstm` - Disable ConvLSTM temporal modeling
- `--no-residual` - Disable residual connections
- `--no-batchnorm` - Disable batch normalization

**Alternative: Environment Variables**
```bash
MS_SSIM_WEIGHT=0.0 USE_CONVLSTM=false ./train_model.sh convlstm_latent_split l1
```

## Overview

The training pipeline supports ablations for both **loss components** and **model architecture features**. You can disable any component by setting its weight to 0 (for losses) or using the appropriate flag (for model features).

## Loss Ablations

Control the contribution of different loss terms:

### MS-SSIM Loss
- **Parameter**: `--ms-ssim-weight` or `MS_SSIM_WEIGHT`
- **Default**: 0.5
- **Disable**: Set to 0
- **Description**: Multi-scale structural similarity loss for perceptual quality

### Reconstruction Loss (L1 or MSE)
- **Parameter**: `--rec-weight` or `REC_WEIGHT`
- **Default**: 0.5
- **Type**: `--loss-type` (l1 or mse)
- **Disable**: Set to 0
- **Description**: Pixel-wise reconstruction error

### Temporal Smoothness Loss
- **Parameter**: `--temporal-weight` or `TEMPORAL_WEIGHT`
- **Default**: 0.1
- **Disable**: Set to 0
- **Description**: Encourages smooth latent transitions between frames

## Model Architecture Ablations

Control key architectural components:

### ConvLSTM
- **Parameter**: `--no-convlstm` or `USE_CONVLSTM=false`
- **Default**: Enabled
- **Description**: Temporal modeling with ConvLSTM. When disabled, no temporal processing occurs.

### Residual Connections
- **Parameter**: `--no-residual` or `USE_RESIDUAL=false`
- **Default**: Enabled
- **Description**: ResNet-style skip connections in encoder/decoder blocks

### Batch Normalization
- **Parameter**: `--no-batchnorm` or `USE_BATCHNORM=false`
- **Default**: Enabled
- **Description**: Batch normalization in all convolutional blocks

### Dropout
- **Parameter**: `--dropout-rate` or `DROPOUT_RATE`
- **Default**: 0.1
- **Disable**: Set to 0
- **Description**: Dropout applied before latent compression

## Usage Examples

### Using Python Command Line

#### Baseline (all features enabled)
```bash
python train.py convlstm_latent_split \
  --loss-type l1 \
  --ms-ssim-weight 0.5 \
  --rec-weight 0.5 \
  --temporal-weight 0.1 \
  --dropout-rate 0.1
```

#### Ablate MS-SSIM Loss
```bash
python train.py convlstm_latent_split \
  --loss-type l1 \
  --ms-ssim-weight 0.0 \
  --rec-weight 1.0 \
  --temporal-weight 0.1
```

#### Ablate Temporal Smoothness Loss
```bash
python train.py convlstm_latent_split \
  --loss-type l1 \
  --ms-ssim-weight 0.5 \
  --rec-weight 0.5 \
  --temporal-weight 0.0
```

#### Ablate ConvLSTM (no temporal modeling)
```bash
python train.py convlstm_latent_split \
  --loss-type l1 \
  --no-convlstm
```

#### Ablate Residual Connections
```bash
python train.py convlstm_latent_split \
  --loss-type l1 \
  --no-residual
```

#### Ablate Batch Normalization
```bash
python train.py convlstm_latent_split \
  --loss-type l1 \
  --no-batchnorm
```

#### Ablate Dropout
```bash
python train.py convlstm_latent_split \
  --loss-type l1 \
  --dropout-rate 0.0
```

#### Multiple Ablations (no temporal modeling, no batch norm)
```bash
python train.py convlstm_latent_split \
  --loss-type l1 \
  --no-convlstm \
  --no-batchnorm \
  --temporal-weight 0.0
```

### Using Shell Script

The `train_model.sh` script accepts ablation parameters directly as command-line arguments:

**Syntax**: `./train_model.sh <MODE> <LOSS_TYPE> [ablation arguments...]`

#### Baseline
```bash
./train_model.sh convlstm_latent_split l1
```

#### Ablate MS-SSIM Loss
```bash
./train_model.sh convlstm_latent_split l1 --ms-ssim-weight 0.0 --rec-weight 1.0
```

#### Ablate ConvLSTM
```bash
./train_model.sh convlstm_latent_split l1 --no-convlstm
```

#### Ablate Residual Connections and Batch Norm
```bash
./train_model.sh convlstm_latent_split l1 --no-residual --no-batchnorm
```

#### Multiple Ablations
```bash
./train_model.sh convlstm_latent_split mse \
  --ms-ssim-weight 0.0 \
  --temporal-weight 0.0 \
  --no-convlstm
```

#### All Loss Weights Custom
```bash
./train_model.sh convlstm_latent_split l1 \
  --ms-ssim-weight 0.3 \
  --rec-weight 0.6 \
  --temporal-weight 0.05 \
  --dropout-rate 0.2
```

### Alternative: Using Environment Variables

You can also use environment variables (useful for batch scripts):

#### Ablate MS-SSIM Loss
```bash
MS_SSIM_WEIGHT=0.0 REC_WEIGHT=1.0 ./train_model.sh convlstm_latent_split l1
```

#### Ablate ConvLSTM
```bash
USE_CONVLSTM=false ./train_model.sh convlstm_latent_split l1
```

#### Multiple Ablations
```bash
MS_SSIM_WEIGHT=0.0 \
TEMPORAL_WEIGHT=0.0 \
USE_CONVLSTM=false \
./train_model.sh convlstm_latent_split mse
```

**Note**: Command-line arguments take precedence over environment variables.

## Reproducibility

All ablation settings are automatically:

1. **Logged to WandB**: Check the `config` section in your WandB run for all hyperparameters
2. **Saved to config files**: `training_config_latent_split.txt` contains full configuration
3. **Stored in the model**: Model objects save ablation parameters for inference

### Config File Locations

- `training_config.txt` - Shell script configuration
- `training_config_latent_split.txt` - Detailed Python training configuration

### WandB Config Fields

All runs include these config fields:
- `loss`: Combined loss description
- `loss_type`: "l1" or "mse"
- `ms_ssim_weight`: MS-SSIM loss weight
- `rec_weight`: Reconstruction loss weight
- `temporal_weight`: Temporal smoothness weight
- `dropout_rate`: Dropout rate
- `use_convlstm`: ConvLSTM enabled/disabled
- `use_residual`: Residual connections enabled/disabled
- `use_batchnorm`: Batch normalization enabled/disabled
- `model_features`: Summary of enabled features

## Recommended Ablation Studies

### Loss Component Analysis
1. **MS-SSIM only**: `ms_ssim_weight=1.0`, `rec_weight=0.0`, `temporal_weight=0.0`
2. **Reconstruction only**: `ms_ssim_weight=0.0`, `rec_weight=1.0`, `temporal_weight=0.0`
3. **No temporal**: `ms_ssim_weight=0.5`, `rec_weight=0.5`, `temporal_weight=0.0`
4. **Baseline**: `ms_ssim_weight=0.5`, `rec_weight=0.5`, `temporal_weight=0.1`

### Architecture Component Analysis
1. **No ConvLSTM**: `--no-convlstm` (tests importance of temporal modeling)
2. **No Residual**: `--no-residual` (tests importance of skip connections)
3. **No BatchNorm**: `--no-batchnorm` (tests importance of normalization)
4. **No Dropout**: `--dropout-rate 0.0` (tests importance of regularization)
5. **Minimal architecture**: All above combined
6. **Baseline**: All features enabled (default)

## Notes

- Setting both `ms_ssim_weight` and `rec_weight` to 0 will result in no reconstruction loss (not recommended)
- Disabling ConvLSTM removes all temporal modeling - frames are processed independently
- Disabling residual connections may hurt gradient flow in deep networks
- Disabling batch normalization may require adjusting learning rate
- All ablation experiments should use the same random seed for fair comparison (set via `PYTHONHASHSEED`, `torch.manual_seed()`, etc.)
