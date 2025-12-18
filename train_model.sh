#!/bin/bash

# Parse arguments or use defaults
MODE="${1:-convlstm}"  # Default to convlstm (options: convlstm, convlstm_latent_split)
LOSS_TYPE="${2:-l1}"  # Default to l1
TEMPORAL_SMOOTHNESS="${3:-true}"  # Default to true

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
Training Configuration
======================
Date: $(date)
Script: train_model.sh
Mode: $MODE
Loss Type: $LOSS_TYPE
Temporal Smoothness: $TEMPORAL_SMOOTHNESS
Latent Split: $([ "$MODE" = "convlstm_latent_split" ] && echo "ENABLED (2048 empty + 2048 embryo)" || echo "DISABLED")
Command: python train.py $MODE --loss-type $LOSS_TYPE $([ "$TEMPORAL_SMOOTHNESS" = "false" ] && echo "--no-temporal-smoothness" || echo "")
EOF

echo "========================================="
echo "Training Configuration:"
echo "  Mode: $MODE"
echo "  Loss Type: $LOSS_TYPE"
echo "  Temporal Smoothness: $TEMPORAL_SMOOTHNESS"
if [ "$MODE" = "convlstm_latent_split" ]; then
    echo "  Latent Split: ENABLED"
    echo "    - Empty Well Latent: 2048 (first half)"
    echo "    - Embryo Latent: 2048 (second half)"
else
    echo "  Latent Split: DISABLED"
fi
echo "========================================="
cat training_config.txt
echo "========================================="

# Run training with specified configuration
if [ "$TEMPORAL_SMOOTHNESS" = "false" ]; then
    python train.py "$MODE" --loss-type "$LOSS_TYPE" --no-temporal-smoothness
else
    python train.py "$MODE" --loss-type "$LOSS_TYPE"
fi

#python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py

rm -r embryo_dataset
