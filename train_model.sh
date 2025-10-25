#!/bin/bash
python -m venv .venv

source .venv/bin/activate
pip install -r requirements.txt
pwd
ls
#python train.py $(realpath ./../../../staging/groups/bhaskar_group/ivf-data)
