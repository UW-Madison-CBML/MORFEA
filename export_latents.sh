#!/bin/bash
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"
echo "Starting export_latents job..."

# Install dependencies
echo "Installing dependencies..."
pip install safetensors huggingface_hub wandb

# Extract dataset
echo "Extracting dataset..."
tar -zxf embryo_dataset.tar.gz
tar -zxf latents.tar.gz
# Set HuggingFace token from api_keys.txt
if [ -f "api_keys.txt" ]; then
    HF_KEY=$(head -n 1 api_keys.txt)
    export HF_TOKEN=$HF_KEY
    echo "HuggingFace token loaded from api_keys.txt"
fi

# Run export script
echo "Running export_latents.py..."
python export_latents.py
mv *.csv latents/
mv *.npy latents/
tar -czvf latents latents.tar.gz
# Cleanup
echo "Cleaning up..."
rm -r embryo_dataset

echo "Export complete!" 

