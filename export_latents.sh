#!/bin/bash
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"
tar -zxf embryo_dataset.tar.gz
python build_index.py
python export_latents.py 

