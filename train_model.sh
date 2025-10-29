#!/bin/bash
python -m venv venv

source train_venv/bin/activate
pip install -r train_requirements.txt
cat train_requirements.txt
pip freeze 
tar -zxf embryo_dataset.tar.gz
python build_index.py
python train.py 


rm -r embryo_dataset
cp *.pth /home/jlundsgaard/ivf
