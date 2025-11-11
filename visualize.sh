#!/bin/bash
mkdir plots
python visualize.py latents.csv
tar -czvf plots.tar.gz plots
