This readme is super WIP. Just writing some notes about dependencies and stuff
# Notes:
* Make sure you have the data downloaded and in the following folders/ files: 
    * embryo\_dataset 
    * embryo\_dataset\_annotations
    * embryo\_dataset/*  
    * embryo\_dataset\_annotations/*
    * embryo\_dataset\_grades.csv 
* If running on mac/windows just install all the necessary dependencies with pip install
* If you clone to linux you'll need to create a .venv environment, run "source .venv/bin/activate", and install the dependencies there
* Current .venv dependencies: torchsummary, natsort, torch (PyTorch), tphate, scipy,pygsp,tasklogger,scikit-learn, future, statsmodels, scprep, s\_gd2, matplotlib
* Reminder: Get a Pipfile set up to do the above
* Most of the other dependencies are just installed globally nothing crazy, numpy and stuff
* (note so I don't forget) If you want to connect git to github (i.e. only have access to a command line), you create a ssh key, add the public key to github, and set up remotes for the repo as "git remote add origin git@github.com:JensLundsgaard/ivf.git". To find where git looks for the private key run "ssh -vT git@github.com"
# TODO:
* Mess with the model a ton, try different activiation functions, different RNN models, different models entirely,
* Incorporate grading\ timestamps of embryos into both model and visualization.
* Start to incorporate TDA 
