#!/bin/bash


curl https://zenodo.org/records/7912264/files/embryo_dataset_grades.csv?download=1 -o embryo_dataset_grades.csv
curl https://zenodo.org/records/7912264/files/embryo_dataset_time_elapsed.tar.gz?download=1 -o embryo_dataset_time_elapsed.tar.gz
curl https://zenodo.org/records/7912264/files/embryo_dataset_annotations.tar.gz?download=1 -o embryo_dataset_annotations.tar.gz

curl https://zenodo.org/records/7912264/files/embryo_dataset.tar.gz?download=1 -o embryo_dataset.tar.gz

#gzip -d embryo_dataset_time_elapsed.tar.gz
#gzip -d embryo_dataset_annotations.tar.gz
#gzip -d embryo_dataset.tar.gz

tar -xf embryo_dataset_time_elapsed.tar
tar -xf embryo_dataset_annotations.tar
tar -xf embryo_dataset.tar.gz
