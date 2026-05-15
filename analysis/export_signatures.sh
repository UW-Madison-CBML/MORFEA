#!/bin/bash
tar -zxf signatures.tar.gz
tar -zxf latents.tar.gz
mkdir -p signatures
ls -lh
python export_signatures.py --name "$1"

tar -czvf signatures.tar.gz signatures
