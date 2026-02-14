# Next Steps After Fixing run_train.sh

## Step 1: Verify Job Status

```bash
condor_q
```

Check if job status is:
- **I** (Idle) = Waiting for H200 GPU (normal, wait)
- **R** (Running) = Training in progress! ✅
- **H** (Held) = Still has issue, check again

## Step 2: Monitor Training (If Running)

```bash
# Replace <cluster.proc> with your actual job ID
condor_tail <cluster.proc>
```

You should see:
- Environment setup
- Dependency installation
- Dataset linking
- Training progress (epochs, loss values)

## Step 3: Check for Errors

```bash
condor_tail -stderr <cluster.proc>
```

If you see errors, common issues:
- Missing `dataset_ivf.py` - wrapper should find it automatically
- Missing `index.csv` - wrapper should build it
- Out of memory - may need to increase `request_memory`

## Step 4: Verify Training is Working

Look for these in the output:
```
Starting training...
Epoch 1/50: 100%|████████| 125/125 [15:23<00:00, loss=0.8234]
Epoch 2/50: 100%|████████| 125/125 [14:56<00:00, loss=0.7891]
...
```

## Step 5: Check Logs (After Some Progress)

```bash
# View output log
tail -n 50 logs/train_<cluster>_<proc>.out

# View error log
tail -n 50 logs/train_<cluster>_<proc>.err
```

## If Job is Still Held

### Check Hold Reason Again

```bash
condor_q -hold
condor_q -long <cluster.proc> | grep -i hold
```

### Common Issues and Fixes

**1. Still "Transfer output files failure"**
- Check if `run_train.sh` was updated correctly
- Verify file permissions: `chmod +x run_train.sh`

**2. "Failed to transfer input files"**
- Verify all files exist: `ls -lh train.py model.py conv_lstm.py losses.py run_train.sh`
- Check file permissions

**3. "Container pull failed"**
- Try different PyTorch image tag
- Or remove container requirement temporarily

**4. "Out of memory"**
- Increase `request_memory = 32GB` in submit file

## Expected Training Timeline

- **Job submission to start**: 0-30 minutes (waiting for H200)
- **Per epoch**: 10-30 minutes
- **Total (50 epochs)**: 8-25 hours
- **Checkpoints**: Saved every 5 epochs

## Quick Status Check Commands

```bash
# All in one
condor_q && echo "---" && condor_q -hold && echo "---" && tail -n 20 logs/train_*.out 2>/dev/null | tail -n 10
```

## After Training Completes

```bash
# Check results location
ls -lh /staging/groups/bhaskar_group/ivf/results/

# Extract and view
cd /staging/groups/bhaskar_group/ivf/results/
tar -xzf results_<cluster>_<proc>.tgz
ls -lh
```





