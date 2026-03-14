#!/bin/bash
pip install iisignature
tar -zxf signatures.tar.gz
tar -zxf latents.tar.gz
mkdir -p signatures
ls -lh
python export_signatures.py --name control-2026-03-12

tar -czvf signatures.tar.gz signatures
