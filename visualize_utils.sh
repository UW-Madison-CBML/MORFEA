#!/bin/bash
mkdir grades
python visualize_utils.py latents.csv --output grades --compare-grades --grades-file embryo_dataset_grades.csv --plot-type grid
tar -cvf grades.tar.gz grades
