# Check Why Job is Held

## Quick Commands

```bash
# 1. 查看被 hold 的任務
condor_q -hold

# 2. 查看 hold 原因（替換為你的 job ID）
condor_q -long <cluster.proc> | grep -i hold

# 3. 查看錯誤日誌
condor_tail -stderr <cluster.proc>

# 4. 查看完整狀態
condor_q -better-analyze <cluster.proc>
```

## Common Hold Reasons

### 1. Transfer Output Files Failure
**Error**: `HoldReason = "Transfer output files failure"`

**Solution**: Check if `results.tgz` is being created properly in `run_train.sh`

### 2. Missing Input Files
**Error**: `HoldReason = "Error from slot1_*: Failed to transfer input files"`

**Solution**: Verify all files in `transfer_input_files` exist

### 3. Container/Image Issues
**Error**: `HoldReason = "Error from slot1_*: Failed to pull container"`

**Solution**: Container image might be unavailable, try different tag

### 4. Permission Issues
**Error**: `HoldReason = "Error from slot1_*: Permission denied"`

**Solution**: Check file permissions, ensure scripts are executable

## Fix Hold Issues

### Release and Resubmit
```bash
# Release hold (if it's a temporary issue)
condor_release <cluster.proc>

# Or remove and resubmit
condor_rm <cluster.proc>
# Fix the issue
condor_submit train_h200_lab.sub
```





