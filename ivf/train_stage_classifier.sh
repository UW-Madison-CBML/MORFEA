#!/bin/bash
echo "Installing dependencies..."
pip install safetensors wandb
 
# Extract dataset
echo "Extracting dataset..."
tar -zxf latents.tar.gz
tar -zxf embryo_dataset_annotations.tar.gz
# Set HuggingFace token from api_keys.txt
if [ -f "api_keys.txt" ]; then
    WB_KEY=$(tail -n 1 api_keys.txt)
    export WANDB_KEY=$WB_KEY
    echo "HuggingFace token loaded from api_keys.txt"
fi
ls -lh

 
python train_stage_classifier.py --name "$1"
