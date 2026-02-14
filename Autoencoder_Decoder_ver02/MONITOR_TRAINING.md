# Monitor Training on CHTC

## After Submitting Job

### Step 1: Check Job Status

```bash
condor_q
```

You should see something like:
```
OWNER BATCH_NAME      SUBMITTED   DONE  RUN IDLE TOTAL JOB_IDS
rho9  ID: 2609123     11/19 15:30  _    _    1     1  2609123.0
```

**Status meanings:**
- **I** (Idle) = Waiting for H200 GPU to become available
- **R** (Running) = Training in progress! 🎉
- **H** (Held) = Error occurred, check with `condor_q -hold`

### Step 2: Get Your Job ID

From `condor_q` output, note the job ID (e.g., `2609123.0`)

### Step 3: Monitor Live Output

```bash
# Replace 2609123.0 with your actual job ID
condor_tail 2609123.0
```

This shows real-time training output. Press `Ctrl+C` to exit.

### Step 4: Check for Errors

```bash
condor_tail -stderr 2609123.0
```

### Step 5: Detailed Status (if job is Idle)

If job stays Idle for a while:

```bash
condor_q -better-analyze 2609123.0
```

This shows why the job isn't matching to a GPU.

### Step 6: Check Log Files (After Job Starts)

```bash
# View output log
tail -f logs/train_2609123_0.out

# View error log
tail -f logs/train_2609123_0.err

# View HTCondor log
tail -f logs/train_2609123_0.log
```

## While Training is Running

### Check Training Progress

The training script will output:
- Epoch progress
- Loss values (L1, MS-SSIM, total)
- Learning rate
- Estimated time remaining

### Expected Output

```
Epoch 1/50: 100%|████████| 125/125 [15:23<00:00, loss=0.8234]
Epoch 2/50: 100%|████████| 125/125 [14:56<00:00, loss=0.7891]
...
```

## After Training Completes

### Step 1: Check Job Status

```bash
condor_q
# Should show job as completed (not listed, or in condor_history)
```

### Step 2: View Final Logs

```bash
# View complete output
cat logs/train_2609123_0.out

# View any errors
cat logs/train_2609123_0.err
```

### Step 3: Check Results Location

Results are automatically saved to staging:

```bash
ls -lh /staging/groups/bhaskar_group/ivf/results/
```

You should see:
```
results_2609123_0.tgz
```

### Step 4: Extract Results (Optional)

```bash
cd /staging/groups/bhaskar_group/ivf/results/
tar -xzf results_2609123_0.tgz
ls -lh
```

Should contain:
- `checkpoints/` - Model checkpoints (every 5 epochs)
- `logs/` - Training logs and metrics JSON

### Step 5: Check Training Metrics

```bash
# View training log JSON
cat logs/training_log.json | python3 -m json.tool
```

## Useful Commands

### Cancel Job (if needed)

```bash
condor_rm 2609123.0
```

### View Job History

```bash
condor_history
```

### Check GPU Usage (if job is running)

```bash
condor_q -long 2609123.0 | grep -i gpu
```

### View Complete Job Info

```bash
condor_q -long 2609123.0
```

## Troubleshooting

### Job Stays Idle

```bash
# Check why
condor_q -better-analyze 2609123.0

# Common reasons:
# - No H200 GPUs available (wait or check with lab)
# - Missing lab attributes (already configured)
```

### Job Fails Immediately

```bash
# Check error log
condor_tail -stderr 2609123.0

# Common issues:
# - Missing dataset_ivf.py (wrapper should find it)
# - Missing index.csv (wrapper should build it)
# - Out of memory (increase request_memory)
```

### Training Crashes Mid-Way

```bash
# Check output for error messages
tail -n 200 logs/train_2609123_0.out

# Check if any checkpoints were saved
ls -lh checkpoints/
```

## Expected Training Timeline

- **Job submission to start**: 0-30 minutes (waiting for H200)
- **Per epoch**: 10-30 minutes
- **Total training (50 epochs)**: 8-25 hours
- **Checkpoints saved**: Every 5 epochs

## Next Steps After Training

1. **Load trained model**:
   ```python
   import torch
   checkpoint = torch.load('checkpoints/checkpoint_epoch_50.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   ```

2. **Extract latents for T-PHATE/TDA**:
   ```python
   output = model(x)
   z_seq = output["z_seq"]   # (B, T, 256, 16, 16)
   z_last = output["z_last"] # (B, 256, 16, 16)
   ```

3. **Analyze results**:
   - Check training curves in `logs/training_log.json`
   - Visualize reconstructions
   - Run T-PHATE on latents





