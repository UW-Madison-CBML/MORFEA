#!/bin/bash
pip install huggingface_hub wandb
HF_KEY=$(head -n 1 api_keys.txt)
export HF_TOKEN=$HF_KEY
WANDB_KEY=$(tail -n 1 api_keys.txt)
export WANDB_KEY=$WANDB_KEY
tar -zxf embryo_dataset.tar.gz

#python train.py 
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py

rm -r embryo_dataset
