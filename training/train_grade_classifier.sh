#!/bin/bash

tar -zxf latents.tar.gz
tar -zxf cebra_latents.tar.gz
tar -zxf kanakasabapathy_latents.tar.gz

# Set HuggingFace token from api_keys.txt
if [ -f "api_keys.txt" ]; then
    WB_KEY=$(tail -n 1 api_keys.txt)
    export WANDB_KEY=$WB_KEY
    echo "HuggingFace token loaded from api_keys.txt"
fi
ls -lh

python train_grade_classifier.py "$@"

