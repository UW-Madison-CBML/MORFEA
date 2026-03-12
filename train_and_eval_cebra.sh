#!/bin/bash
python -m ruff check . --select F821,E9 || exit 1
mkdir cebra_plots

tar -xf embryo_dataset.tar.gz
tar -xf embryo_dataset_annotations.tar.gz
HF_KEY=$(head -n 1 api_keys.txt)
export HF_TOKEN=$HF_KEY

python train_and_eval_cebra.py "$1"

tar -czvf cebra_plots.tar.gz cebra_plots/
