#!/bin/bash
python -m venv venv
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"
source train_venv/bin/activate
pip install -r train_requirements.txt
cat train_requirements.txt
pip freeze 
tar -zxf embryo_dataset.tar.gz
python build_index.py
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024"

python train.py 


rm -r embryo_dataset
cp *.pth /home/jlundsgaard/ivf
