#!/bin/bash
echo "Starting export_latents job..."
# Install dependencies
echo "Installing dependencies..."
pip install safetensors huggingface_hub wandb

# Extract dataset
echo "Extracting dataset..."
tar -zxf embryo_dataset.tar.gz
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

for i in "${ADDR[@]}"; do
	    python export_latents.py --name "$i" --limit 50
done

#cat get_latents.txt | xargs -I {} sh -c 'python export_latents.py --name "{}" --limit 50'
mkdir "${1}"_latents
mv *.csv "${1}"_latents
mv *.npy "${1}"_latents
tar -czvf "${1}"_latents.tar.gz "${1}"_latents
# Cleanup
echo "Cleaning up..."
rm -r embryo_dataset

echo "Export complete!" 
