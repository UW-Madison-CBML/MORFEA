# Fix results.tgz Transfer Failure

## Problem
Task failed during training, so `results.tgz` was never created. HTCondor tried to transfer it but failed, causing the job to be held.

## Root Cause
Training script failed (likely couldn't find `dataset_ivf.py` or `index.csv`), so `run_train.sh` never reached the part that creates `results.tgz`.

## Solution

### Step 1: Check why training failed

```bash
cat logs/train_2650361_0.err
tail -n 100 logs/train_2650361_0.out
```

### Step 2: Fix run_train.sh to always create results.tgz

The script should create `results.tgz` even if training fails. Update it:

```bash
cd ~/ivf_train

# Download updated run_train.sh from GitHub
curl -L -o run_train.sh "https://raw.githubusercontent.com/UW-Madison-CBML/ivf/main/Raffael/2025-11-19/run_train.sh"

# Or manually fix it to ensure results.tgz is always created
```

### Step 3: Fix the actual training issue

Most likely issues:
1. **Missing dataset_ivf.py** - script can't find it
2. **Missing index.csv** - can't build or find it
3. **Wrong data path** - `/project/bhaskar_group/ivf` doesn't exist or wrong structure

### Step 4: Resubmit

After fixing issues:
```bash
condor_rm 2650361.0
# Fix issues
condor_submit train_h200_lab.sub
```





