#!/bin/bash
tar -zxf embryo_dataset.tar.gz

python train.py 

rm -r embryo_dataset
