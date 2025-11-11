*This readme is super WIP. Just writing some notes about dependencies and stuff*

# Notes:
* [Instructions for getting this repo onto CHTC](https://docs.google.com/document/d/1x7FlTtsYuOqeQj37U9IaIt8Nphn9OEqrRj0oT5o2rCc/edit?usp=sharing)
* We have a [Docker image](https://hub.docker.com/r/jenslundsgaard/ivf-training/tags) now!!!! It's `jenslundsgaard/ivf-training:first`. It contains all of the dependencies for training and using the model on GPUs. 
* We also have a [tphate image](https://hub.docker.com/r/jenslundsgaard/tphate/tags) now. This whole repo is designed to be ran on docker please use one of the above. 
* DO NOT change `train_requirements.txt`. It is currently what loads our deps.
* Make sure you have `alias python=python3` in your .bashrc file.
* Also make sure you run `chmod +x SCRIPT.sh` so that bash has permission to run the script
* (note so I don't forget) If you want to connect git to github (i.e. only have access to a command line), you create a ssh key, add the public key to github, and set up remotes for the repo as `git remote add origin git@github.com:JensLundsgaard/ivf.git`. To find where git looks for the private key run `ssh -vT git@github.com`
* For copying a file from a running docker container do `sudo docker cp 897f96a9e24e:/app/plots/tphate_scatter.png - | sudo -u $USER tee tphate_scatter.png > /dev/null`
# TODO:
* Mess with the model a ton, try different activiation functions, different RNN models, different models entirely,
* Incorporate grading\ timestamps of embryos into both model and visualization.
* Start to incorporate TDA 
