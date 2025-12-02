#!/bin/bash
tar -xvzf embryo_dataset_annotations.tar.gz
mkdir grades
python visualize_utils.py latents.csv --output grades --compare-grades --grades-file embryo_dataset_grades.csv --plot-type grid --coloring phase
python visualize_utils.py latents.csv --output grades --compare-grades --grades-file embryo_dataset_grades.csv --plot-type grid --coloring velocity
python visualize_utils.py latents.csv --output grades --compare-grades --grades-file embryo_dataset_grades.csv --plot-type grid --coloring curvature

tar -cvf grades.tar.gz grades
