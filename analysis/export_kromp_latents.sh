#!/bin/bash

tar -xvf Blastocyst_Dataset.tar.gz
ls Blastocyst_Dataset/

mkdir -p kromp_latents

if [ -f "api_keys.txt" ]; then
    HF_KEY=$(head -n 1 api_keys.txt)
    export HF_TOKEN=$HF_KEY
fi
python export_kromp_latents.py --name "$1"

mv "$1".npy kromp_latents
mv "$1".csv kromp_latents
tar -czvf kromp_latents.tar.gz kromp_latents/

