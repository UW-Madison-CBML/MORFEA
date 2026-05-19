#!/bin/bash

tar -zxf embryo_dataset.tar.gz

if [ -f "api_keys.txt" ]; then
    WB_KEY=$(tail -n 1 api_keys.txt)
    export WANDB_KEY=$WB_KEY
    echo "HuggingFace token loaded from api_keys.txt"
fi
ls -lh

python train_lstm_classifier.py 

