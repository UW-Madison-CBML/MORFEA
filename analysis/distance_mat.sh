#!/bin/bash
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"
mkdir -p "$1"_distances
tar -xvf latents.tar.gz
tar -xvf embryo_dataset_annotations.tar.gz
mkdir "$1"_imgs
python distance_mat.py --name "$1"
ls -lh "$1"_distances | wc -l
tar -czvf "$1"_distances.tar.gz "$1"_distances

