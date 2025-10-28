#!/bin/bash
python -m venv venv

source train_venv/bin/activate
pip install -r train_requirements.txt
tar -zxf embryo_dataset.tar.gz
ls -a
pip freeze
python build_index.py
python train.py 


rm -r embryo_dataset
cp *.pth /home/jlundsgaard/ivf
