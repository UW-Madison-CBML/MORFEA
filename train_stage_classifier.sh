#!/bin/bash
python -m ruff check . --exit-zero
echo "Installing dependencies..."
# Extract dataset
echo "Extracting dataset..."
tar -zxf latents.tar.gz
tar -zxf embryo_dataset_annotations.tar.gz
pip install pytorch-crf
# Set HuggingFace token from api_keys.txt
if [ -f "api_keys.txt" ]; then
    WB_KEY=$(tail -n 1 api_keys.txt)
    export WANDB_KEY=$WB_KEY
    echo "HuggingFace token loaded from api_keys.txt"
fi
ls -lh

 
python "$@"
