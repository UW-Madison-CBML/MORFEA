#!/bin/bash

echo ""

PROJECT_DIR="/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"
cd "$PROJECT_DIR"

files=(
    "extract_all_latent_trajectories.py"
    "model.py"
    "dataset_ivf.py"
    "build_index.py"
    "extract_latents.sh"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file - MISSING!"
        exit 1
    fi
done

echo ""
if grep -q "read directly from CSV" extract_all_latent_trajectories.py; then
else
    exit 1
fi

echo ""
if grep -q "tar_file" dataset_ivf.py && grep -q "tarfile.open" dataset_ivf.py; then
else
    exit 1
fi

echo ""
if head -1 extract_latents.sh | grep -q "#!/bin/bash"; then
else
    exit 1
fi

if grep -q "需要我提供" extract_latents.sh; then
    exit 1
else
fi

echo ""

echo ""

