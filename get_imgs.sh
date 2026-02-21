#!/bin/bash
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"
pip install safetensors huggingface_hub
tar -zxf embryo_dataset.tar.gz
tar -zxf embryo_dataset_annotations.tar.gz
mkdir "$1"_imgs
pip install huggingface_hub wandb safetensors
HF_KEY=$(head -n 1 api_keys.txt)
export HF_TOKEN=$HF_KEY
echo "$1"
python get_img.py --name "$1"
tar -czvf "$1"_imgs.tar.gz "$1"_imgs

