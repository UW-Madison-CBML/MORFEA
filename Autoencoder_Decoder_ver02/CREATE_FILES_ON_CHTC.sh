#!/bin/bash
# Create all files directly on CHTC (no download needed)
# Copy and paste this entire script into CHTC terminal

cd ~/ivf_train

echo "=== Creating ConvLSTM Autoencoder Files on CHTC ==="
echo ""

# 1. Create train_h200_lab.sub
echo "1. Creating train_h200_lab.sub..."
cat > train_h200_lab.sub << 'SUBEOF'
executable = run_train.sh
arguments  =

log    = logs/train_$(Cluster)_$(Process).log
output = logs/train_$(Cluster)_$(Process).out
error  = logs/train_$(Cluster)_$(Process).err

transfer_input_files = run_train.sh, train.py, model.py, conv_lstm.py, losses.py

+WantGPULab = true
+GPUJobLength = "short"
gpus_minimum_capability = 7.0

+ProjectName = "UWMadison_BME_Bhaskar"

+SingularityImage = "docker://pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"

request_gpus   = 1
request_cpus   = 2
request_memory = 16GB
request_disk   = 30GB

Requirements = regexp("H200", TARGET.CUDADeviceName)

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_output_files   = results.tgz
transfer_output_remaps  = "results.tgz = file:///staging/groups/bhaskar_group/ivf/results/results_$(Cluster)_$(Process).tgz;"

queue 1
SUBEOF

echo "✅ train_h200_lab.sub created"
echo ""

# 2. Create run_train.sh
echo "2. Creating run_train.sh..."
cat > run_train.sh << 'RUNEOF'
#!/bin/bash
set -e

# Use shared dataset
ln -sfn /project/bhaskar_group/ivf data

# Install dependencies
PYDEPS="$PWD/pydeps"
mkdir -p "$PYDEPS"
python3 -m pip install --no-cache-dir --upgrade pip
python3 -m pip install --no-cache-dir opencv-python pandas numpy tqdm scikit-learn matplotlib ripser persim -t "$PYDEPS"
export PYTHONPATH="$PYDEPS:$PYTHONPATH"

# Add dataset_ivf to path
if [ -f "../../Code/dataset_ivf.py" ]; then
    export PYTHONPATH="../../Code:$PYTHONPATH"
elif [ -f "../../../Code/dataset_ivf.py" ]; then
    export PYTHONPATH="../../../Code:$PYTHONPATH"
fi

# Build index if needed
if [ ! -f index.csv ]; then
    if [ -f "../../Code/build_index.py" ]; then
        python3 ../../Code/build_index.py
    fi
fi

# Run training
python3 train.py \
    --index_csv index.csv \
    --batch_size 8 \
    --seq_len 20 \
    --num_epochs 50 \
    --learning_rate 3e-4 \
    --save_dir checkpoints \
    --log_dir logs

# Package results (CRITICAL: must create results.tgz)
if [ -d "checkpoints" ] && [ -d "logs" ]; then
    tar -czf results.tgz checkpoints/ logs/ 2>/dev/null || tar -czf results.tgz logs/
elif [ -d "logs" ]; then
    tar -czf results.tgz logs/
elif [ -d "checkpoints" ]; then
    tar -czf results.tgz checkpoints/
else
    echo "empty" > empty.txt && tar -czf results.tgz empty.txt && rm -f empty.txt
fi

# Verify results.tgz exists
if [ ! -f results.tgz ]; then
    echo "empty" > empty.txt && tar -czf results.tgz empty.txt && rm -f empty.txt
fi

echo "Results packaged: $(ls -lh results.tgz)"
RUNEOF

chmod +x run_train.sh
echo "✅ run_train.sh created"
echo ""

# 3. Verify files
echo "3. Verifying files..."
ls -lh train_h200_lab.sub run_train.sh

# Check if Python files exist
if [ ! -f "train.py" ] || [ ! -f "model.py" ] || [ ! -f "conv_lstm.py" ] || [ ! -f "losses.py" ]; then
    echo ""
    echo "⚠️  Python files (train.py, model.py, conv_lstm.py, losses.py) are missing!"
    echo "   You need to download them from GitHub or upload from your Mac"
    echo ""
    echo "Download command:"
    echo "  BASE_URL=\"https://raw.githubusercontent.com/UW-Madison-CBML/ivf/main/Raffael/2025-11-19\""
    echo "  curl -L -o train.py \"\${BASE_URL}/train.py\""
    echo "  curl -L -o model.py \"\${BASE_URL}/model.py\""
    echo "  curl -L -o conv_lstm.py \"\${BASE_URL}/conv_lstm.py\""
    echo "  curl -L -o losses.py \"\${BASE_URL}/losses.py\""
else
    echo "✅ All Python files present"
fi

echo ""
echo "=== Ready to Submit ==="
echo ""
echo "Next step:"
echo "  mkdir -p logs"
echo "  condor_submit train_h200_lab.sub"
echo "  condor_q"





