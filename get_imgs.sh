#!/bin/bash
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"
tar -zxf embryo_dataset.tar.gz
python build_index.py
rm -r imgs 
rm imgs.tar.gz
mkdir imgs
python get_img.py 
tar -czvf imgs.tar.gz imgs

rm -r embryo_dataset
cp *.pth /home/jlundsgaard/ivf
