#!/bin/bash
set -euo pipefail

echo "Installing dependencies..."
pip install -q safetensors wandb scikit-learn umap-learn matplotlib

echo "Extracting dataset..."
tar -zxf latents_for_staging.tar.gz
tar -zxf cebra_latents.tar.gz
tar -zxf embryo_dataset_annotations.tar.gz

if [ -f "api_keys.txt" ]; then
    export WANDB_KEY="$(tail -n 1 api_keys.txt)"
    echo "WANDB_KEY loaded from api_keys.txt"
else
    export WANDB_MODE=disabled
    echo "No api_keys.txt — WandB disabled (training still runs)."
fi

ls -lh latents/ 2>/dev/null | head || true
ls -lh cebra_latents/ 2>/dev/null | head || true
ls utils/ 2>/dev/null || true

python "$@"
