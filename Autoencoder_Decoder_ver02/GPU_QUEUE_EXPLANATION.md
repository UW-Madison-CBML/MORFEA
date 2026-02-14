# GPU Queue and Lab Priority Explanation

## How Lab Settings Work

### What `+WantGPULab = true` Does:
- Tells HTCondor you want to use the **lab's GPU pool**
- Matches you to lab-specific GPU nodes (like `bhaskargpu4000.chtc.wisc.edu`)
- **Does NOT guarantee immediate access** - you still wait in queue

### What `+ProjectName = "UWMadison_BME_Bhaskar"` Does:
- Identifies your job as part of Bhaskar lab
- Helps with accounting and resource allocation
- May give priority within lab's allocation

## Queue Behavior

### Normal Queue:
- Jobs wait for available GPUs
- First-come-first-served within same priority
- Lab jobs may have higher priority than general pool

### Your Situation:
- H200 GPUs are limited (only 8 in your lab)
- Other lab members may be using them
- Job will start when:
  1. H200 GPU becomes available
  2. Your job is highest priority in queue
  3. Job requirements match (H200, lab pool, etc.)

## Expected Wait Time

- **Best case**: 0-30 minutes (if GPU available)
- **Typical**: 1-4 hours (if others using GPUs)
- **Worst case**: 8+ hours (if all GPUs busy)

## How to Check Queue Status

```bash
# Check your job position
condor_q -better-analyze <job_id>

# Check available H200 GPUs
condor_status -constraint 'regexp("H200", CUDADeviceName)' -af Name State Activity

# Check lab GPU usage
condor_status -constraint 'regexp("bhaskar", Machine)' -af Name TotalGPUs GPUs
```

## Tips to Reduce Wait Time

1. **Submit during off-peak hours** (evenings, weekends)
2. **Use shorter job length** (already set: `+GPUJobLength = "short"`)
3. **Check with lab members** if GPUs are available
4. **Consider H100 as backup** (if acceptable):
   ```bash
   # Modify Requirements line:
   Requirements = regexp("(H200|H100)", TARGET.CUDADeviceName)
   ```

## Current Status

Your job is correctly configured for lab priority:
- ✅ `+WantGPULab = true` - Uses lab GPU pool
- ✅ `+ProjectName = "UWMadison_BME_Bhaskar"` - Lab identification
- ✅ `+GPUJobLength = "short"` - Shorter jobs get priority
- ✅ `gpus_minimum_capability = 7.0` - Matches H200

Just need to wait for H200 to become available!





