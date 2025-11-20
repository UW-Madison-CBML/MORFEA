#!/bin/bash
# Training wrapper script for CHTC
# Uses shared dataset and new ConvLSTM Autoencoder model

set -e

# Use shared dataset
ln -sfn /project/bhaskar_group/ivf data

# Install dependencies into local directory (container may not have all packages)
PYDEPS="$PWD/pydeps"
mkdir -p "$PYDEPS"
python3 -m pip install --no-cache-dir --upgrade pip
python3 -m pip install --no-cache-dir opencv-python pandas numpy tqdm scikit-learn matplotlib ripser persim -t "$PYDEPS"
export PYTHONPATH="$PYDEPS:$PYTHONPATH"

# Build index if needed (try to find build_index.py)
if [ ! -f index.csv ]; then
    echo "Building index.csv..."
    # Try multiple possible locations for build_index.py
    if [ -f "../../Code/build_index.py" ]; then
        python3 ../../Code/build_index.py
    elif [ -f "../../../Code/build_index.py" ]; then
        python3 ../../../Code/build_index.py
    else
        echo "build_index.py not found, will use existing index.csv or create during training"
    fi
fi

# Run training
echo "Starting training..."
python3 train.py \
    --index_csv index.csv \
    --batch_size 8 \
    --seq_len 20 \
    --num_epochs 50 \
    --learning_rate 3e-4 \
    --save_dir checkpoints \
    --log_dir logs

# Package results
echo "Packaging results..."
tar -czf results.tgz checkpoints/ logs/ 2>/dev/null || tar -czf results.tgz logs/ 2>/dev/null

echo "Training completed!"

