#!/bin/bash
pip install safetensors huggingface_hub wandb
pip install transformers torch
pip install --upgrade transformers
# Extract dataset
python -c "import transformers; print(transformers.__version__)"
tar -zxf embryo_dataset.tar.gz
tar -zxf latents.tar.gz
tar -zxf embryo_dataset_annotations.tar.gz
if [ -f "api_keys.txt" ]; then
    HF_KEY=$(head -n 1 api_keys.txt)
    export HF_TOKEN=$HF_KEY
    echo "HuggingFace token loaded from api_keys.txt"
fi
ls -lh
echo "Running export_latents.py..."
IFS="_" read -ra ADDR <<< "$1"

python femi_eval.py 

#cat get_latents.txt | xargs -I {} sh -c 'python export_latents.py --name "{}" --limit 50'
mkdir -p latents
mv *.npy latents/
mv *.csv latents/
tar -I 'gzip -1' -cf latents.tar.gz latents/

