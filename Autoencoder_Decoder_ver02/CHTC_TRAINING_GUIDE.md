# CHTC Training Guide

## Quick Start

### 1. SSH to CHTC Access Point

```bash
ssh rho9@ap2001.chtc.wisc.edu
```

### 2. Clone Repository and Setup

```bash
cd ~
git clone https://github.com/UW-Madison-CBML/ivf.git
cd ivf/Raffael/2025-11-19

# Run setup script
bash chtc_train_setup.sh
```

Or manually:

```bash
# Clone
cd ~
git clone https://github.com/UW-Madison-CBML/ivf.git
cd ivf/Raffael/2025-11-19

# Create venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch opencv-python pandas numpy tqdm scikit-learn matplotlib ripser persim
```

### 3. Prepare Training Scripts

The `run_train.sh` wrapper is already configured to:
- Use shared dataset: `/project/bhaskar_group/ivf`
- Install missing dependencies
- Run training with proper parameters
- Package results

### 4. Create Submit File

```bash
cat > train_h200_lab.sub << 'EOF'
executable = run_train.sh
arguments  =

log    = logs/train_$(Cluster)_$(Process).log
output = logs/train_$(Cluster)_$(Process).out
error  = logs/train_$(Cluster)_$(Process).err

transfer_input_files = \
    run_train.sh, \
    train.py, \
    model.py, \
    conv_lstm.py, \
    losses.py

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

mkdir -p logs
chmod +x run_train.sh
```

### 5. Submit Job

```bash
condor_submit train_h200_lab.sub
condor_q
```

### 6. Monitor Training

```bash
# View job status
condor_q

# View live output
condor_tail <cluster.proc>

# View logs after completion
tail -n 100 logs/train_<cluster>_<proc>.out
```

## File Structure on CHTC

```
~/ivf/Raffael/2025-11-19/
├── conv_lstm.py
├── model.py
├── losses.py
├── train.py
├── test_model.py
├── verify_data_connection.py
├── run_train.sh
├── train_h200_lab.sub
├── .venv/          (created by setup)
└── logs/           (created by submit)
```

## Important Notes

1. **Dataset**: Uses shared dataset at `/project/bhaskar_group/ivf`
2. **Container**: PyTorch 2.4.0 with CUDA 12.1
3. **Dependencies**: Installed in job (not in container)
4. **Results**: Saved to `/staging/groups/bhaskar_group/ivf/results/`
5. **H200 Only**: Strictly requires H200 GPU

## Troubleshooting

If job is Idle:
- Check: `condor_q -better-analyze <cluster.proc>`
- May need to add `+AccountingGroup` or `+ProjectName` if required

If training fails:
- Check: `condor_tail -stderr <cluster.proc>`
- Verify dataset path is correct
- Check that index.csv exists or can be built





