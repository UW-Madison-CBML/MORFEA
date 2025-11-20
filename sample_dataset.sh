#!/bin/bash

# Extract the embryo dataset
echo "Extracting embryo_dataset.tar.gz..."
tar -zxf embryo_dataset.tar.gz

# Create sample directory
mkdir -p embryo_dataset_sample

# Select 25 random subfolders and copy them
echo "Selecting 25 random subfolders..."
ls -d embryo_dataset/*/ | sed 's|embryo_dataset/||g' | sed 's|/||g' | shuf | head -25 | while read folder; do
    echo "  Adding $folder"
    cp -r "embryo_dataset/$folder" "embryo_dataset_sample/"
done

# Clean up original and rename sample
rm -r embryo_dataset
mv embryo_dataset_sample embryo_dataset

# Create compressed archive
echo "Creating embryo_dataset_sample.tar.gz..."
tar -czf embryo_dataset_sample.tar.gz embryo_dataset
