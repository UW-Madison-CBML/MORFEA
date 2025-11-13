#!/bin/bash

# Extract the embryo dataset
echo "Extracting embryo_dataset.tar.gz..."
tar -zxf embryo_dataset.tar.gz

# Get the total number of files
total_files=$(find embryo_dataset -type f | wc -l)
echo "Total files in dataset: $total_files"

# Calculate 5% of files
sample_size=$(echo "$total_files * 0.05" | bc | awk '{printf "%d", $1}')
echo "Sampling 5% ($sample_size files)..."

# Create a temporary directory for the sample
mkdir -p embryo_dataset_sample

# Get a random 5% sample of files and copy them
find embryo_dataset -type f | shuf | head -n $sample_size | while read file; do
    # Preserve directory structure
    relative_path=${file#embryo_dataset/}
    mkdir -p "embryo_dataset_sample/$(dirname "$relative_path")"
    cp "$file" "embryo_dataset_sample/$relative_path"
done

# Create compressed archive of the sample
echo "Creating embryo_dataset_sample.tar.gz..."
tar -czf embryo_dataset_sample.tar.gz embryo_dataset_sample

# Cleanup
echo "Cleaning up..."
rm -rf embryo_dataset embryo_dataset_sample

echo "Done! Created embryo_dataset_sample.tar.gz"
