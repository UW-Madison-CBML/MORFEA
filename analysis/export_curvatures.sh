#!/bin/bash
pip install --no-cache-dir umap-learn[tbb]
tar -zxf curvatures.tar.gz
tar -zxf latents.tar.gz
mkdir -p curvatures
ls -lh
python export_curvatures.py --name control-2026-01-06

tar -czvf signatures.tar.gz signatures
