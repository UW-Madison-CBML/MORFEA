#!/bin/bash

source .venv/bin/activate

pip install -r requirements.txt

python train.py $(realpath ./../../../staging/groups/bhaskar_group/ivf-data)
