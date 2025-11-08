This readme is super WIP. Just writing some notes about dependencies and stuff
# Notes:
* Make sure you have the data downloaded and in the following folders/ files: 
```
embryo_dataset 
embryo_dataset_annotations
embryo_dataset/*  
embryo_dataset_annotations/*
embryo_dataset_grades.csv 
```
* Currently due to some weird stuff with pytorch, `model_weights.pth` can only be loaded in the container `docker_image = pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime`. To make sure the model loads well I'm setting up the job `docker_test.sub` to make sure the model weights file in the group shared folder will have the right shape when you actually run a job on it.
* If you just cloned the repo to linux run `chmod +x add_venvs.sh` and then `./add_venvs`. This will create 3 different venvs for different use cases. `train_venv` is for training/ building the data index etc. `opencv_venv` is for using the `opencv-python` library. `tphate_venv` is for visualization and results purposes. I set this up to help deal with dependency conflicts. 
* If you add any other dependencies please run `pip freeze > $VENV_requirements.txt` where `$VENV` is the virtual environment name.
* Make sure you have `alias python=python3` in your .bashrc file.
* Also make sure you run `chmod +x SCRIPT.sh` so that bash has permission to run the script
* (note so I don't forget) If you want to connect git to github (i.e. only have access to a command line), you create a ssh key, add the public key to github, and set up remotes for the repo as `git remote add origin git@github.com:JensLundsgaard/ivf.git`. To find where git looks for the private key run `ssh -vT git@github.com`
# TODO:
* Get docker workin
* Mess with the model a ton, try different activiation functions, different RNN models, different models entirely,
* Incorporate grading\ timestamps of embryos into both model and visualization.
* Start to incorporate TDA 
