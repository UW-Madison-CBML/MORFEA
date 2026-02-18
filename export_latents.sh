#!/bin/bash
echo "Starting export_latents job..."
# Install dependencies
echo "Installing dependencies..."
pip install safetensors huggingface_hub wandb

# Extract dataset
echo "Extracting dataset..."
tar -zxf embryo_dataset.tar.gz
tar -zxf latents.tar.gz
tar -zxf embryo_dataset_annotations.tar.gz
# Set HuggingFace token from api_keys.txt
if [ -f "api_keys.txt" ]; then
    HF_KEY=$(head -n 1 api_keys.txt)
    export HF_TOKEN=$HF_KEY
    echo "HuggingFace token loaded from api_keys.txt"
fi
ls -lh
# Run export script
echo "Running export_latents.py..."
IFS="_" read -ra ADDR <<< "$1"

#for i in "${ADDR[@]}"; do
#done
python export_latents.py --name control-2026-02-16 
python export_latents.py --name noconv-2026-02-16 
python export_latents.py --name notemp-2026-02-16 

#cat get_latents.txt | xargs -I {} sh -c 'python export_latents.py --name "{}" --limit 50'
# Combine into one pass, using pigz for speed
tar -I 'gzip -1' -cf latents.tar.gz latents/

# Cleanup
echo "Cleaning up..."
rm -r embryo_dataset

echo "Export complete!" 
