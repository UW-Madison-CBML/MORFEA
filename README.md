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
* If you add any other dependencies please run `pip freeze > requirements.txt`
* Make sure you have `alias python=python3` in your .bashrc file.
* Also make sure you run `chmod +x SCRIPT.sh` so that bash has permission to run the script
* (note so I don't forget) If you want to connect git to github (i.e. only have access to a command line), you create a ssh key, add the public key to github, and set up remotes for the repo as `git remote add origin git@github.com:JensLundsgaard/ivf.git`. To find where git looks for the private key run `ssh -vT git@github.com`
# TODO:
* Fix python dependency compatability issues. Currenly `pip freeze > requirements.txt` and `pip install -r requirements.txt` is a bit annoying as workflow since compatability is not taken into account. Would like to switch to something that will install all necessary versions when you build it. Take a look at this error:
```
scprep 1.2.3 requires pandas<2.1,>=0.25, but you have pandas 2.3.3 which is incompatible.
opencv-python 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 2.3.4 which is incompatible.
```
`scprep` is a requirement for tphate. I think the solution right now is to just have multiple venv environments. I'll add a bash script that will install these.
* Get docker workin
* Mess with the model a ton, try different activiation functions, different RNN models, different models entirely,
* Incorporate grading\ timestamps of embryos into both model and visualization.
* Start to incorporate TDA 
