# MORFEA
Recurrent encoders can preserve morphodynamic features.
## Introduction
The goal of this project is to develop a method for studying human in-vitro fertilized embryos using unsupervised deep learning. We use a convolutional recurrent autoencoder to embed embryo image sequences and then study the geometric and morphodynamic features of these *latent trajectories*.

## Tutorial
### Dependencies 
All core machine learning code is ran in the Docker image `jenslundsgaard/ivf-training:v3.3`. Some downstream tasks are ran in a venv environment with `requirements.txt` installed with pip.
### Download
Clone the repository with 
```
git clone https://github.com/UW-Madison-CBML/MORFEA.git
```
### Data Preparation
We use two datasets: Gomez at al. (also called the video dataset) [here](https://zenodo.org/records/7912264) and Kanakasabapthy et al. (also called the single frame dataset) [here](https://osf.io/3kc2d/overview). From the video dataset you will need to download the files:
```
embryo_dataset_grades.csv
embryo_dataset.tar.gz
embryo_dataset_annotation.tar.gz
```
Unzip these and have the folders in the directory where you are training/ running scripts. From the single frame dataset download `ed4.zip` from the Embryo dataset, and pull out `alldata`. The directories inside this folder should have the names 1,2,3,4,5. Rename the folder to `kanakasabapathy/` and move it to the directory where you are running scripts.

### Use the pretrained model
To use the model with pretrained weights, simply run
```
model = ConvGRUAutoencoder.from_pretrained('JensLundsgaard/convlstm_final-2026-07-13')
```
### Build the data indexes
To build the indexes `utils/index.csv` and `utils/index_embryo.csv`, run
```
python build_index.py
```
and
```
python build_index_embryo.py
```
### Train a new model 
To train the model from scratch, go to the `training/` folder, and run 
``` 
python train_ae.py convgru --name NEW_MODEL [--ARGS]
 ```
For a description of arguments, see the end of `train_ae.py`, and the files `ablation.txt`, `train_ae.sub` and `train_ae.sh`. **Note**: the code is setup to run on our compute system which grabs all necessary files. You may need to add `sys.path.append("...")` to make sure all necessary files can be seen by the training script. For a list of such files, see the line `transfer_input_files=` in `train_ae.sub`.
### WandB and Hugging Face logging
To log any runs you do with the autoencoder or downstream tasks, set the file `api_keys.txt` to 
```
HF_KEY
WANDB_KEY
```
See `train.sh` for how to add these as environment variables which the scripts can access.
### Exporting Latents
Once the model is trained and pushed to a Hugging Face repo, move to the `analysis/` folder, and run 
```
python export_video_latents.py --name NEW_MODEL-DATE
```
then run 
```
python export_kanakasabapthy_latents.py --name NEW_MODEL-DATE
``` 
```
python export_cebra.py --name NEW_MODEL-DATE
``` 
You may need to edit the Hugging Face repository you are pulling from to run the above.
### Downstream Tasks
Once you have exported the model's latents, you can try downstream tasks. These include stage and grade prediction, as well as visualizing the latents in 3d PCA space. For stage and grade prediction, go to the `training/` folder and then run 
```
python train_stage_classifier.py --model-name NEW_MODEL-DATE [--ARGS]
```   
```
python train_grade_classifier.py --kanakasabapathy --model-name NEW_MODEL-DATE [--ARGS]
```
or 
```
python train_grade_classifier.py --model-name NEW_MODEL-DATE [--ARGS]
```
for grade classification. See the respective .py, .sh, .sub, and .txt files for more information on arguments.


