#!/bin/bash
#python -m ruff check . --select F821,E9 || exit 1
#mkdir -p cebra_plots
#tar -xf embryo_dataset_annotations.tar.gz

tar -xf latents.tar.gz
tar -xf cebra_latents.tar.gz
tar -xf cebra_plots.tar.gz

python visualize_cebra.py "$@"


tar -czvf cebra_plots.tar.gz cebra_plots/
