#!/bin/bash
echo "Installing dependencies..."
pip install safetensors wandb scikit-learn seaborn
mkdir clusters
# Extract dataset
echo "Extracting dataset..."
tar -zxf signatures.tar.gz
# Set HuggingFace token from api_keys.txt
if [ -f "api_keys.txt" ]; then
    WB_KEY=$(tail -n 1 api_keys.txt)
    export WANDB_KEY=$WB_KEY
    echo "HuggingFace token loaded from api_keys.txt"
fi
ls -lh
# Run export script
echo "Running export_latents.py..."

python cluster_signatures.py --name "$1"
tar -czvf clusters.tar.gz clusters
