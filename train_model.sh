#!/bin/bash
python -m venv .venv

source .venv/bin/activate
pip install -r requirements.txt

curl https://zenodo.org/records/7912264/files/embryo_dataset.tar.gz?download=1 -o embryo_dataset.tar.gz
gzip -d embryo_dataset.tar.gz
ls -a
tar -xf embryo_dataset.tar
ls -a
python build_index.py
python train.py 

cp model_weights.pth /home/jlundsgaard/ivf
