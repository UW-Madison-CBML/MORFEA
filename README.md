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
* I am trying to get CHTC figured out. I will probably set it up so the train.py can take a realpath as input and will expect to find the above dirs at the given location.
* If you just cloned the repo to linux you'll need to create a .venv environment with `python -m venv .venv`, run `source .venv/bin/activate`, and run `pip install -r requirements.txt` and it will install. If you have any other notes about getting the code to work on CHTC, let me know or add to this file. 
* Make sure you have `alias python=python3` in your .bashrc file.
* (note so I don't forget) If you want to connect git to github (i.e. only have access to a command line), you create a ssh key, add the public key to github, and set up remotes for the repo as `git remote add origin git@github.com:JensLundsgaard/ivf.git`. To find where git looks for the private key run `ssh -vT git@github.com`
# TODO:
* Mess with the model a ton, try different activiation functions, different RNN models, different models entirely,
* Incorporate grading\ timestamps of embryos into both model and visualization.
* Start to incorporate TDA 
