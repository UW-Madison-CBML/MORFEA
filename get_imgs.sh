#!/bin/bash
python -m venv venv
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"
source venv/bin/activate
pip install -r train_requirements.txt
tar -zxf embryo_dataset.tar.gz
python build_index.py
mkdir imgs
python get_img.py 
tar -czvf imgs.tar.gz /imgs

rm -r embryo_dataset
cp *.pth /home/jlundsgaard/ivf
