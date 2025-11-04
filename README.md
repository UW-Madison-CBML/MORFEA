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
* In terms of docker I'm not really sure what to set up right now; using it with chtc is very easy, its just the top two lines of `train_model.sub`
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
