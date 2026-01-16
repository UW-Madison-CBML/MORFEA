#!/bin/bash

# Parse positional arguments
MODE="${1:-convlstm_latent_split}"  # Default to convlstm_latent_split
LOSS_TYPE="${2:-l1}"  # Default to l1

# Shift past the first two arguments to get ablation parameters
shift 2 2>/dev/null || true

# Default values (can be overridden by environment variables or command-line args)
MS_SSIM_WEIGHT="${MS_SSIM_WEIGHT:-0.5}"
REC_WEIGHT="${REC_WEIGHT:-0.5}"
TEMPORAL_WEIGHT="${TEMPORAL_WEIGHT:-0.1}"
DROPOUT_RATE="${DROPOUT_RATE:-0.1}"
USE_CONVLSTM="${USE_CONVLSTM:-true}"
USE_RESIDUAL="${USE_RESIDUAL:-true}"
USE_BATCHNORM="${USE_BATCHNORM:-true}"

# Parse command-line arguments (override defaults and environment variables)
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --ms-ssim-weight)
            MS_SSIM_WEIGHT="$2"
            shift 2
            ;;
        --rec-weight)
            REC_WEIGHT="$2"
            shift 2
            ;;
        --temporal-weight)
            TEMPORAL_WEIGHT="$2"
            shift 2
            ;;
        --dropout-rate)
            DROPOUT_RATE="$2"
            shift 2
            ;;
        --no-convlstm)
            USE_CONVLSTM="false"
            shift
            ;;
        --no-residual)
            USE_RESIDUAL="false"
            shift
            ;;
        --no-batchnorm)
            USE_BATCHNORM="false"
            shift
            ;;
        *)
            # Pass through any unrecognized arguments
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

pip install huggingface_hub wandb safetensors
HF_KEY=$(head -n 1 api_keys.txt)
export HF_TOKEN=$HF_KEY
WANDB_KEY=$(tail -n 1 api_keys.txt)
export WANDB_KEY=$WANDB_KEY
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
tar -zxf embryo_dataset.tar.gz

# Create training configuration file for reproducibility
cat > training_config.txt << EOF
ABLATION STUDY - Training Configuration
========================================
Date: $(date)
Script: train_model.sh
Mode: $MODE

Loss Configuration:
  - Loss Type: $LOSS_TYPE
  - MS-SSIM Weight: $MS_SSIM_WEIGHT $([ "$MS_SSIM_WEIGHT" = "0" ] && echo "(DISABLED)" || echo "")
  - Reconstruction Weight: $REC_WEIGHT $([ "$REC_WEIGHT" = "0" ] && echo "(DISABLED)" || echo "")
  - Temporal Weight: $TEMPORAL_WEIGHT $([ "$TEMPORAL_WEIGHT" = "0" ] && echo "(DISABLED)" || echo "")

Model Architecture:
  - ConvLSTM: $([ "$USE_CONVLSTM" = "true" ] && echo "ENABLED" || echo "DISABLED")
  - Residual Connections: $([ "$USE_RESIDUAL" = "true" ] && echo "ENABLED" || echo "DISABLED")
  - Batch Normalization: $([ "$USE_BATCHNORM" = "true" ] && echo "ENABLED" || echo "DISABLED")
  - Dropout Rate: $DROPOUT_RATE $([ "$DROPOUT_RATE" = "0" ] && echo "(DISABLED)" || echo "")

Latent Split: $([ "$MODE" = "convlstm_latent_split" ] && echo "ENABLED (2048 empty + 2048 embryo)" || echo "DISABLED")

Command: python train.py $MODE \\
  --loss-type $LOSS_TYPE \\
  --ms-ssim-weight $MS_SSIM_WEIGHT \\
  --rec-weight $REC_WEIGHT \\
  --temporal-weight $TEMPORAL_WEIGHT \\
  --dropout-rate $DROPOUT_RATE \\
  $([ "$USE_CONVLSTM" = "false" ] && echo "--no-convlstm" || echo "") \\
  $([ "$USE_RESIDUAL" = "false" ] && echo "--no-residual" || echo "") \\
  $([ "$USE_BATCHNORM" = "false" ] && echo "--no-batchnorm" || echo "")
EOF

echo "========================================="
echo "ABLATION STUDY - Training Configuration"
echo "========================================="
echo ""
echo "Mode: $MODE"
echo ""
echo "Loss Configuration:"
echo "  - Loss Type: $LOSS_TYPE"
echo "  - MS-SSIM Weight: $MS_SSIM_WEIGHT $([ "$MS_SSIM_WEIGHT" = "0" ] && echo "(DISABLED)" || echo "")"
echo "  - Reconstruction Weight: $REC_WEIGHT $([ "$REC_WEIGHT" = "0" ] && echo "(DISABLED)" || echo "")"
echo "  - Temporal Weight: $TEMPORAL_WEIGHT $([ "$TEMPORAL_WEIGHT" = "0" ] && echo "(DISABLED)" || echo "")"
echo ""
echo "Model Architecture:"
echo "  - ConvLSTM: $([ "$USE_CONVLSTM" = "true" ] && echo "ENABLED" || echo "DISABLED")"
echo "  - Residual Connections: $([ "$USE_RESIDUAL" = "true" ] && echo "ENABLED" || echo "DISABLED")"
echo "  - Batch Normalization: $([ "$USE_BATCHNORM" = "true" ] && echo "ENABLED" || echo "DISABLED")"
echo "  - Dropout Rate: $DROPOUT_RATE $([ "$DROPOUT_RATE" = "0" ] && echo "(DISABLED)" || echo "")"
echo ""
if [ "$MODE" = "convlstm_latent_split" ]; then
    echo "Latent Split: ENABLED"
    echo "  - Empty Well Latent: 2048 (first half)"
    echo "  - Embryo Latent: 2048 (second half)"
else
    echo "Latent Split: DISABLED"
fi
echo "========================================="
cat training_config.txt
echo "========================================="

# Build command with ablation arguments
CMD="python train.py $MODE --loss-type $LOSS_TYPE"
CMD="$CMD --ms-ssim-weight $MS_SSIM_WEIGHT"
CMD="$CMD --rec-weight $REC_WEIGHT"
CMD="$CMD --temporal-weight $TEMPORAL_WEIGHT"
CMD="$CMD --dropout-rate $DROPOUT_RATE"

if [ "$USE_CONVLSTM" = "false" ]; then
    CMD="$CMD --no-convlstm"
fi

if [ "$USE_RESIDUAL" = "false" ]; then
    CMD="$CMD --no-residual"
fi

if [ "$USE_BATCHNORM" = "false" ]; then
    CMD="$CMD --no-batchnorm"
fi

# Add any extra arguments that were passed through
if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

# Run training with specified configuration
echo "Executing: $CMD"
eval $CMD

#python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py

rm -r embryo_dataset
