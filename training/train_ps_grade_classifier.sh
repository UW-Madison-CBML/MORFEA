#!/bin/bash

python -m ruff check . --select F821,E9 || exit 1

HF_KEY=$(head -n 1 api_keys.txt)
export HF_TOKEN=$HF_KEY

WANDB_KEY=$(tail -n 1 api_keys.txt)
export WANDB_KEY=$WANDB_KEY


tar -xvf latents.tar.gz

python train_ps_grade_classifier.py "$@"

