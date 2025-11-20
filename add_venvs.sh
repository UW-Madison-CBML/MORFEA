#!/bin/bash
# building data index and training the model
python3 -m venv train_venv

source train_venv/bin/activate

pip install -r train_requirements.txt

deactivate
# visualizing, getting results or plotting with TPHATE and similar tools
python3 -m venv tphate_venv

source tphate_venv/bin/activate

pip install -r tphate_requirements.txt

deactivate
# using opencv for cleaning data, etc
python3 -m venv opencv_venv

source opencv_venv/bin/activate

pip install -r opencv_requirements.txt

deactivate

