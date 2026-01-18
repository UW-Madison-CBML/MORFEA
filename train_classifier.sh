#!/bin/bash
echo "Installing dependencies..."
pip install safetensors wandb

# Extract dataset
echo "Extracting dataset..."
tar -zxf signatures.tar.gz
# Set HuggingFace token from api_keys.txt
if [ -f "api_keys.txt" ]; then
    HF_KEY=$(head -n 1 api_keys.txt)
    export HF_TOKEN=$HF_KEY
    echo "HuggingFace token loaded from api_keys.txt"
fi
ls -lh
# Run export script
echo "Running export_latents.py..."

python train_classifier.py --name "$1"

