#!/bin/bash
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"
pip install safetensors huggingface_hub
tar -zxf embryo_dataset.tar.gz
rm -r imgs 
rm imgs.tar.gz
mkdir imgs
pip install huggingface_hub wandb safetensors
HF_KEY=$(head -n 1 api_keys.txt)
export HF_TOKEN=$HF_KEY

python get_img.py --name JensLundsgaard/embryo-convlstm-ls-084cf5d1-2025-12-23
tar -czvf imgs.tar.gz imgs

rm -r embryo_dataset
cp *.pth /home/jlundsgaard/ivf
