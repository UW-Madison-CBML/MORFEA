#!/bin/bash
pip install safetensors huggingface_hub
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"
tar -zxf embryo_dataset.tar.gz
python export_latents.py 

