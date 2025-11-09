#!/bin/bash
tar -zxf embryo_dataset.tar.gz
python build_index.py

python train.py 

rm -r imgs
rm imgs.tar.gz
mkdir imgs
python get_img.py 
tar -czvf imgs.tar.gz ./imgs


rm -r embryo_dataset
