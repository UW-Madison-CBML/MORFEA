#!/bin/bash
tar -xvzf embryo_dataset_annotations.tar.gz
tar -xvf latents.tar.gz
mkdir -p curvate_plots

python visualize_utils.py "$1" --output grades --compare-grades --grades-file embryo_dataset_grades.csv --plot-type grid --coloring phase 

tar -cvf "$1"_curvature_plots.tar.gz curvature_plots

