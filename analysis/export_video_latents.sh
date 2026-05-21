#!/bin/bash
echo "Starting export_video_latents job..."

# Extract dataset
echo "Extracting dataset..."
tar -zxf embryo_dataset.tar.gz
tar -zxf latents.tar.gz
tar -zxf embryo_dataset_annotations.tar.gz
# Set HuggingFace token from api_keys.txt
if [ -f "api_keys.txt" ]; then
    HF_KEY=$(head -n 1 api_keys.txt)
    export HF_TOKEN=$HF_KEY
    echo "HuggingFace token loaded from api_keys.txt"
fi
ls -lh
# Run export script
echo "Running export_video_latents.py..."
IFS="_" read -ra ADDR <<< "$1"

#for i in "${ADDR[@]}"; do
#done
python export_video_latents.py --name "$1"

#cat get_latents.txt | xargs -I {} sh -c 'python export_video_latents.py --name "{}" --limit 50'
mkdir -p latents
mv *.npy latents/
mv *.csv latents/
tar -I 'gzip -1' -cf latents.tar.gz latents/

