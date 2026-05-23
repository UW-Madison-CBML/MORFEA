#!/bin/bash
#python -m ruff check . --select F821,E9 || exit 1
#mkdir -p cebra_plots
#tar -xf embryo_dataset_annotations.tar.gz

tar -xf latents.tar.gz
tar -xf pca_plots.tar.gz

python visualize_pca.py "$@"


tar -czvf pca_plots.tar.gz pca_plots/
