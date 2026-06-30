#!/bin/bash

python -m ruff check . --select F821,E9 || exit 1

HF_KEY=$(head -n 1 api_keys.txt)
export HF_TOKEN=$HF_KEY

WANDB_KEY=$(tail -n 1 api_keys.txt)
export WANDB_KEY=$WANDB_KEY

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

tar -zxf embryo_dataset.tar.gz

tar -xvf kanakasabapathy.tar.gz
mv alldata/ kanakasabapathy/

python train_ae.py "$@"

#python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py

rm -r embryo_dataset
