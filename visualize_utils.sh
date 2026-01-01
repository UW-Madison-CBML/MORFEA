#!/bin/bash
tar -xvzf embryo_dataset_annotations.tar.gz
tar -xvf latents.tar.gz
mkdir grades

cat visualize_models.txt | xargs -I {} sh -c '
python visualize_utils.py "$1" --output grades --compare-grades --grades-file embryo_dataset_grades.csv --plot-type grid --coloring phase && \
python visualize_utils.py "$1" --output grades --compare-grades --grades-file embryo_dataset_grades.csv --plot-type grid --coloring velocity && \
python visualize_utils.py "$1" --output grades --compare-grades --grades-file embryo_dataset_grades.csv --plot-type grid --coloring curvature
' -- {}

tar -cvf grades.tar.gz grades
