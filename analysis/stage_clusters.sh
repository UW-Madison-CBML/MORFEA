#!/bin/bash
echo "Installing dependencies..."
pip install safetensors wandb
pip install scikit-learn 
pip install "numpy<2.0.0" scikit-learn-extra --upgrade
pip install umap-learn
# Extract dataset
mkdir stage_clusters
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

 
python stage_clusters.py --name "$1"
tar -czvf "$1"_stage_clusters.tar.gz stage_clusters/
