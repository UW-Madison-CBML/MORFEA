#!/bin/bash
tar -xvzf embryo_dataset_annotations.tar.gz
tar -xvf latents.tar.gz
mkdir -p curvate_plots

python visualize_curvature.py "$1"

tar -cvf "$1"_curvature_plots.tar.gz curvature_plots

