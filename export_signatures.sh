#!/bin/bash
tar -zxf signatures.tar.gz
tar -zxf latents.tar.gz

python export_signatures.py --name latents

tar -czvf signatures.tar.gz signatures
