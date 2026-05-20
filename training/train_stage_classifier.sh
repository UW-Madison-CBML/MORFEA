#!/bin/bash
#python -m ruff check . --exit-zero

echo "Extracting dataset..."
tar -zxf latents.tar.gz
ls latents/
tar -zxf cebra_latents.tar.gz
tar -zxf embryo_dataset_annotations.tar.gz
if [ -f "api_keys.txt" ]; then
    WB_KEY=$(tail -n 1 api_keys.txt)
    export WANDB_KEY=$WB_KEY
    echo "HuggingFace token loaded from api_keys.txt"
fi
ls -lh

 
python train_stage_classifier.py "$@"
