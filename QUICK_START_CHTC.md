# Quick Start: Train on CHTC H200

## One-Command Setup and Submit

Run these commands on CHTC access point (ap2001):

```bash
# 1. Clone and setup
cd ~ && \
git clone https://github.com/UW-Madison-CBML/ivf.git && \
cd ivf/Raffael/2025-11-19 && \
python3 -m venv .venv && \
source .venv/bin/activate && \
pip install --upgrade pip && \
pip install torch opencv-python pandas numpy tqdm scikit-learn matplotlib ripser persim

# 2. Create submit file
cat > train_h200_lab.sub << 'EOF'
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
EOF

# 3. Create wrapper script
cat > run_train.sh << 'EOF'
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

# Add dataset_ivf to path (try multiple locations)
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
EOF

chmod +x run_train.sh
mkdir -p logs

# 4. Submit
condor_submit train_h200_lab.sub
condor_q
```

## Monitor Training

```bash
# View status
condor_q

# Live output
condor_tail <cluster.proc>

# After completion
ls -lh /staging/groups/bhaskar_group/ivf/results/
```

