#!/bin/bash

# Extract the embryo dataset
echo "Extracting embryo_dataset.tar.gz..."
tar -zxf embryo_dataset.tar.gz

# Get the total size of the dataset in bytes
total_size=$(du -sb embryo_dataset | awk '{print $1}')
echo "Total dataset size: $(numfmt --to=iec-i --suffix=B $total_size 2>/dev/null || echo $total_size bytes)"

# Calculate 5% of the total size in bytes
target_size=$(echo "scale=2;  $total_size * 0.05" | bc)
echo "Target sample size (5%): $(numfmt --to=iec-i --suffix=B $target_size 2>/dev/null || echo $target_size bytes)"

# Create a temporary directory for the sample
mkdir -p embryo_dataset_sample

# Get list of all folders with their sizes and shuffle
# Format: size folder_name
declare -a folders
declare -a folder_sizes

while IFS=' ' read -r size folder; do
    folders+=("$folder")
    folder_sizes+=("$size")
done < <(du -sb embryo_dataset/*/ | awk '{print $1, $2}' | sed 's|embryo_dataset/||g' | sed 's|/||g' | shuf)

# Accumulate folders until we reach ~5% of total size
accumulated_size=0
selected_count=0

echo "Selecting folders to reach target size..."
for i in "${!folders[@]}"; do
    folder="${folders[$i]}"
    size="${folder_sizes[$i]}"

    # Add this folder to the sample
    echo "  Adding $folder ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo $size bytes))"
    cp -r "embryo_dataset/$folder" "embryo_dataset_sample/"

    accumulated_size=$((accumulated_size + size))
    selected_count=$((selected_count + 1))

    # Stop if we've reached or exceeded the target
    if [[ accumulated_size -gt target_size ]]; then
        break
    fi
done

# Report final statistics
final_size=$(du -sb embryo_dataset_sample | awk '{print $1}')
percentage=$(echo "scale=2; $final_size * 100 / $total_size" | bc)
echo ""
echo "Selected $selected_count folders"
echo "Final sample size: $(numfmt --to=iec-i --suffix=B $final_size 2>/dev/null || echo $final_size bytes) ($percentage%)"
rm -r embryo_dataset
mv embryo_dataset_sample embryo_dataset
# Create compressed archive of the sample
echo "Creating embryo_dataset_sample.tar.gz..."
tar -czf embryo_dataset_sample.tar.gz embryo_dataset


