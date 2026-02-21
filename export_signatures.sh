#!/bin/bash
pip install --no-cache-dir umap-learn[tbb]
tar -zxf signatures.tar.gz
tar -zxf latents.tar.gz
mkdir -p signatures
ls -lh
python export_signatures.py --name control-2026-02-20

tar -czvf signatures.tar.gz signatures
