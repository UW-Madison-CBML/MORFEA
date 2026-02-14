#!/bin/bash
# Copy and paste this entire script into CHTC terminal
# Creates all necessary files directly on CHTC

set -e

echo "=== Creating ConvLSTM Autoencoder files on CHTC ==="

# Create directory
mkdir -p ~/ivf_train
cd ~/ivf_train

echo "Creating files..."

# Create run_train.sh
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

# Package results
tar -czf results.tgz checkpoints/ logs/ 2>/dev/null || tar -czf results.tgz logs/
RUNEOF

# Create train_h200_lab.sub
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

chmod +x run_train.sh
mkdir -p logs

echo "✓ Files created!"
echo ""
echo "Next: Download Python files from GitHub or create them manually"
echo "Files needed: conv_lstm.py, model.py, losses.py, train.py"





