#!/bin/bash
python -m venv .venv

source .venv/bin/activate
pip install -r requirements.txt
# ./download_data.sh
curl https://zenodo.org/records/7912264/files/embryo_dataset.tar.gz?download=1 -o embryo_dataset.tar.gz
tar -xf embryo_dataset.tar.gz
ls -a
pip freeze
python build_index.py
python train.py 

cp *.pth /home/jlundsgaard/ivf
