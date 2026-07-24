#!/bin/bash

tar -xvf kanakasabapathy.tar.gz
mv alldata/ kanakasabapathy/
tar -xvf kanakasabapathy_latents.tar.gz
if [ -f "api_keys.txt" ]; then
    HF_KEY=$(head -n 1 api_keys.txt)
    export HF_TOKEN=$HF_KEY
fi
python export_kanakasabapathy_latents.py --name "$1"

mv "$1".npy kanakasabapathy_latents
mv "$1".csv kanakasabapathy_latents
tar -czvf kanakasabapathy_latents.tar.gz kanakasabapathy_latents/

