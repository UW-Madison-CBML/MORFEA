# Check Job Status Without Staging Access

## If you can't access staging directory

Check these instead:

### 1. Check Job History

```bash
condor_history
```

This shows completed jobs. Look for your job ID and exit code:
- Exit code 0 = Success ✅
- Exit code != 0 = Failed ❌

### 2. Check Local Logs

```bash
# List all log files
ls -lht logs/

# View the most recent output log
cat $(ls -t logs/train_*.out | head -1)

# View the most recent error log
cat $(ls -t logs/train_*.err | head -1)
```

### 3. Check if Training Actually Ran

```bash
# Look for training progress in logs
grep -i "epoch\|loss\|training\|starting" logs/train_*.out | tail -20

# Check if checkpoints were created locally
ls -lh checkpoints/ 2>/dev/null || echo "No checkpoints found"
```

### 4. Check HTCondor Log

```bash
# View the most recent HTCondor log
cat $(ls -t logs/train_*.log | head -1) | tail -50
```

This shows:
- When job started
- When job ended
- Exit code
- Transfer status

### 5. Check Job Details

```bash
# Get your job ID from history first
condor_history | head -1

# Then check details (replace with your job ID)
condor_q -long <cluster.proc> | grep -i "exit\|hold\|transfer\|error"
```

## Quick All-in-One Check

```bash
echo "=== Job History ===" && \
condor_history | head -3 && \
echo "" && \
echo "=== Latest Logs ===" && \
ls -lht logs/ | head -3 && \
echo "" && \
echo "=== Training Output ===" && \
grep -i "epoch\|loss" $(ls -t logs/train_*.out 2>/dev/null | head -1) 2>/dev/null | tail -10 && \
echo "" && \
echo "=== Errors ===" && \
cat $(ls -t logs/train_*.err 2>/dev/null | head -1) 2>/dev/null | tail -10
```





