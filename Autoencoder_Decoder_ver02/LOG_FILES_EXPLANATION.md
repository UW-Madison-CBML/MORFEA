# CHTC Log Files Explanation

## Three Types of Log Files

### 1. Output Log (`train_2650361_0.out`)
**What it is:**
- Your training script's **standard output** (stdout)
- Everything your Python script prints with `print()`
- Training progress, loss values, epoch information

**What you'll see:**
```
Using device: cuda
Loading dataset...
Dataset size: 1234
Initializing model...
Total parameters: 13,451,843
Epoch 1/50: 100%|████████| 125/125 [15:23<00:00, loss=0.8234]
Epoch 2/50: 100%|████████| 125/125 [14:56<00:00, loss=0.7891]
...
```

**When to check:**
- ✅ **Most important** - Check this first!
- See training progress
- Monitor loss values
- Verify training is working

**How to view:**
```bash
cat logs/train_2650361_0.out
tail -f logs/train_2650361_0.out  # Live updates
```

---

### 2. Error Log (`train_2650361_0.err`)
**What it is:**
- Your training script's **standard error** (stderr)
- Python errors, exceptions, tracebacks
- Warnings from libraries
- System errors

**What you'll see (if there are errors):**
```
Traceback (most recent call last):
  File "train.py", line 123, in <module>
    ...
ValueError: ...
```

**When to check:**
- ❌ If training fails or stops unexpectedly
- ⚠️ If output log shows errors
- 🔍 To debug problems

**How to view:**
```bash
cat logs/train_2650361_0.err
# If empty, that's good! (no errors)
```

**Note:** If this file is empty or small, that's **good** - it means no errors!

---

### 3. HTCondor Log (`train_2650361_0.log`)
**What it is:**
- HTCondor's **system log** about your job
- Job lifecycle events (start, end, transfer)
- Resource usage statistics
- File transfer information
- System-level messages

**What you'll see:**
```
000 (2650361.000.000) 2025-11-19 15:50:40 Job submitted
001 (2650361.000.000) 2025-11-19 15:51:34 Job executing on slot1_1@bhaskargpu4000
005 (2650361.000.000) 2025-11-19 15:53:02 Image size: 164044 KB
...
```

**When to check:**
- 🔍 To see when job started/ended
- 📊 To check resource usage
- 🐛 To debug HTCondor issues (not training issues)
- 📦 To verify file transfers

**How to view:**
```bash
cat logs/train_2650361_0.log
tail -n 50 logs/train_2650361_0.log  # Last 50 lines
```

---

## Quick Reference

| Log File | Purpose | Check When | Empty = Good? |
|----------|---------|------------|---------------|
| `.out` | Training output | **Always** | ❌ No (should have content) |
| `.err` | Errors/warnings | When problems occur | ✅ Yes (empty = no errors) |
| `.log` | HTCondor system | Debugging system issues | ❌ No (should have content) |

## Most Important: Output Log

**For monitoring training, check `.out` file:**
```bash
# View current progress
tail -n 50 logs/train_2650361_0.out

# Watch live updates
tail -f logs/train_2650361_0.out
```

## If Training Fails

1. **Check `.err` first** - see what went wrong
2. **Check `.out`** - see what happened before error
3. **Check `.log`** - see if it's a system issue





