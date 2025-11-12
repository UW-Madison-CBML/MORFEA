*This readme is super WIP. Just writing some notes about dependencies and stuff*

# Notes:
* [Instructions for getting this repo onto CHTC](https://docs.google.com/document/d/1x7FlTtsYuOqeQj37U9IaIt8Nphn9OEqrRj0oT5o2rCc/edit?usp=sharing)
* We have a [Docker image](https://hub.docker.com/r/jenslundsgaard/ivf-training/tags) now!!!! It's `jenslundsgaard/ivf-training:first`. It contains all of the dependencies for training and using the model on GPUs. 
* We also have a [TPHATE image](https://hub.docker.com/r/jenslundsgaard/tphate/tags) now. This whole repo is designed to be ran on docker please use one of the above. 
* Check out [Docker Desktop](https://www.docker.com/products/docker-desktop/) if you want to run any code in this repo on a personal computer.
* DO NOT change `train_requirements.txt`. It is currently what loads our deps.
* Make sure you have `alias python=python3` in your .bashrc file.
* Also make sure you run `chmod +x *.sh`
* (note so I don't forget) If you want to connect git to github (i.e. only have access to a command line), you create a ssh key, add the public key to github, and set up remotes for the repo as `git remote add origin git@github.com:JensLundsgaard/ivf.git`. To find where git looks for the private key run `ssh -vT git@github.com`
# TODO:
* Mess with the model a ton, try different activiation functions, different RNN models, different models entirely,
* Incorporate grading\ timestamps of embryos into both model and visualization.
* Start to incorporate TDA 
# Workflows
* **Clear CHTC log, out and err files**: `./clear.sh`
* **Train the model**: `./run.sh` or `condor_submit train_model.sub`
* **Get model latents as csv (stored in group staging)**: `condor_submit export_latents.sub` 
* **Get model latent csv batchs**: `condor_submit get_cell_groups.sub`
* **Get model TPHATE plots (in group storage `ivf/plots.tar.gz`)**: `condor_submit visualize.sub`. The log files will be hidden, run `ls -a` to see them.
