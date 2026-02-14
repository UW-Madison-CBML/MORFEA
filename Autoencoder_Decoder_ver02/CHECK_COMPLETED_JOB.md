# Check Completed/Removed Job

## When condor_q shows 0 jobs

This means the job is no longer in the queue. It could be:
1. ✅ **Completed successfully**
2. ❌ **Failed and exited**
3. 🗑️ **Removed/cancelled**

## Step 1: Check Job History

```bash
condor_history
```

This shows recently completed/removed jobs. Look for your job ID.

## Step 2: Check Results Location

```bash
# Check if results were transferred
ls -lh /staging/groups/bhaskar_group/ivf/results/

# Look for files matching your job ID
ls -lh /staging/groups/bhaskar_group/ivf/results/results_* 2>/dev/null
```

## Step 3: Check Local Logs

```bash
# View all log files
ls -lh logs/

# Check output log (replace with your job ID)
cat logs/train_<cluster>_<proc>.out

# Check error log
cat logs/train_<cluster>_<proc>.err

# Check HTCondor log
cat logs/train_<cluster>_<proc>.log
```

## Step 4: Check if Training Actually Ran

```bash
# Look for training output in logs
grep -i "epoch\|loss\|training" logs/train_*.out | tail -20

# Check if checkpoints were created
ls -lh checkpoints/ 2>/dev/null || echo "No checkpoints directory"
```

## Step 5: Determine What Happened

### If results.tgz exists in staging:
✅ **Job completed successfully!**

```bash
cd /staging/groups/bhaskar_group/ivf/results/
tar -xzf results_<cluster>_<proc>.tgz
ls -lh
```

### If no results but logs show training:
⚠️ **Training may have failed mid-way**

Check error log for specific error:
```bash
cat logs/train_<cluster>_<proc>.err
```

### If logs show immediate failure:
❌ **Job failed to start**

Common causes:
- Missing input files
- Container pull failed
- Permission issues
- Script errors

## Quick Diagnostic Commands

```bash
# All-in-one check
echo "=== Job History ===" && \
condor_history | head -5 && \
echo "" && \
echo "=== Results ===" && \
ls -lh /staging/groups/bhaskar_group/ivf/results/ 2>/dev/null | tail -5 && \
echo "" && \
echo "=== Local Logs ===" && \
ls -lht logs/ | head -5 && \
echo "" && \
echo "=== Last Output ===" && \
tail -n 30 logs/train_*.out 2>/dev/null | tail -n 10
```





