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
cat get_latents.txt | xargs -I {} sh -c 'python export_latents.py --name "{}" --limit 50'
#python export_latents.py --name embryo-convlstm-ls-b882c653-2025-12-30 --limit 50
#python export_latents.py --name embryo-convlstm-ls-4b322aaf-2025-12-29 --limit 50
mv *.csv latents
mv *.npy latents
tar -czvf latents.tar.gz latents
# Cleanup
echo "Cleaning up..."
rm -r embryo_dataset

echo "Export complete!" 

