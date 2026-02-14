# Complete CHTC Training Guide - From Zero

## Step-by-Step: Clone, Setup, and Train on CHTC H200

### Step 1: SSH to CHTC Access Point

```bash
ssh rho9@ap2001.chtc.wisc.edu
```

Enter your password and Duo passcode when prompted.

### Step 2: Clone Repository from GitHub

```bash
cd ~
git clone https://github.com/UW-Madison-CBML/ivf.git
cd ivf/Raffael/2025-11-19
```

### Step 3: Verify Files Are Present

```bash
ls -lh
```

You should see:
- `conv_lstm.py`
- `model.py`
- `losses.py`
- `train.py`
- `run_train.sh`
- `train_h200_lab.sub`
- `README.md`

### Step 4: Make Scripts Executable

```bash
chmod +x run_train.sh
mkdir -p logs
```

### Step 5: Submit Training Job

```bash
condor_submit train_h200_lab.sub
```

You should see output like:
```
Submitting job(s).
1 job(s) submitted to cluster 2609123.
```

### Step 6: Check Job Status

```bash
condor_q
```

You should see your job listed. Status meanings:
- **I** = Idle (waiting for resources)
- **R** = Running (training in progress)
- **H** = Held (error, check with `condor_q -hold`)

### Step 7: Monitor Training (While Running)

Replace `<cluster.proc>` with your actual job ID from `condor_q`:

```bash
# View live output
condor_tail <cluster.proc>

# View errors
condor_tail -stderr <cluster.proc>

# Detailed status
condor_q -better-analyze <cluster.proc>
```

### Step 8: Check Results (After Completion)

```bash
# Results are saved to staging
ls -lh /staging/groups/bhaskar_group/ivf/results/

# Or check local logs
tail -n 100 logs/train_*.out
```

## Complete One-Liner (Copy-Paste Everything)

```bash
cd ~ && \
git clone https://github.com/UW-Madison-CBML/ivf.git && \
cd ivf/Raffael/2025-11-19 && \
chmod +x run_train.sh && \
mkdir -p logs && \
condor_submit train_h200_lab.sub && \
condor_q
```

## Troubleshooting

### Job Stays Idle

```bash
# Check why job isn't matching
condor_q -better-analyze <cluster.proc>
```

Common reasons:
- No H200 GPUs available (wait or check with lab)
- Missing lab-specific attributes (already included in submit file)

### Job Fails

```bash
# Check error log
condor_tail -stderr <cluster.proc>

# Check output log
condor_tail <cluster.proc>
```

Common issues:
- Missing `dataset_ivf.py` - wrapper script should find it automatically
- Missing `index.csv` - wrapper script will try to build it
- Out of memory - increase `request_memory` in submit file

### Check Job History

```bash
condor_history
```

## What Happens During Training

1. **Job starts**: HTCondor finds H200 GPU
2. **Files transferred**: Code files copied to execute node
3. **Container starts**: PyTorch 2.4.0 + CUDA 12.1 container
4. **Environment setup**: Dependencies installed in `pydeps/`
5. **Dataset linked**: Symlink to `/project/bhaskar_group/ivf`
6. **Index built**: `index.csv` created if missing
7. **Training runs**: 50 epochs, batch size 8, seq_len 20
8. **Results packaged**: Checkpoints and logs saved to `results.tgz`
9. **Results transferred**: Saved to `/staging/groups/bhaskar_group/ivf/results/`

## Expected Training Time

- **Per epoch**: ~10-30 minutes (depends on dataset size)
- **Total (50 epochs)**: ~8-25 hours
- **Checkpoints**: Saved every 5 epochs

## Files Created

After training completes:
- `checkpoints/` - Model checkpoints (every 5 epochs)
- `logs/` - Training logs and metrics
- `results.tgz` - Packaged results (transferred to staging)

## Next Steps After Training

1. **Extract results**:
   ```bash
   cd /staging/groups/bhaskar_group/ivf/results/
   tar -xzf results_<cluster>_<proc>.tgz
   ```

2. **Use latents for T-PHATE/TDA**:
   - Load model checkpoint
   - Extract `z_seq` and `z_last` from trained model
   - Run T-PHATE or TDA analysis





