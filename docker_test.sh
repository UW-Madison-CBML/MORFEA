#!/bin/bash

python -m venv venv

source venv/bin/activate
pip freeze
pip install -r train_requirements.txt

python model.py model_weights.pth
