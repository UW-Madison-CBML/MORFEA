#!/bin/bash

# Parse arguments or use defaults
LOSS_TYPE="${1:-l1}"  # Default to l1
TEMPORAL_SMOOTHNESS="${2:-true}"  # Default to true

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
Loss Type: $LOSS_TYPE
Temporal Smoothness: $TEMPORAL_SMOOTHNESS
Command: python train.py convlstm --loss-type $LOSS_TYPE $([ "$TEMPORAL_SMOOTHNESS" = "false" ] && echo "--no-temporal-smoothness" || echo "")
EOF

echo "========================================="
echo "Training Configuration:"
echo "  Loss Type: $LOSS_TYPE"
echo "  Temporal Smoothness: $TEMPORAL_SMOOTHNESS"
echo "========================================="
cat training_config.txt
echo "========================================="

# Run training with specified configuration
if [ "$TEMPORAL_SMOOTHNESS" = "false" ]; then
    python train.py convlstm --loss-type "$LOSS_TYPE" --no-temporal-smoothness
else
    python train.py convlstm --loss-type "$LOSS_TYPE"
fi

#python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py

rm -r embryo_dataset
