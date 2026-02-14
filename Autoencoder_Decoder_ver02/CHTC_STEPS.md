# CHTC Training Steps - Execute on CHTC

## Step-by-Step Instructions

### Step 1: SSH to CHTC

```bash
ssh rho9@ap2001.chtc.wisc.edu
```

### Step 2: Clone Repository

```bash
cd ~
git clone https://github.com/UW-Madison-CBML/ivf.git
cd ivf/Raffael/2025-11-19
```

### Step 3: Setup Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch opencv-python pandas numpy tqdm scikit-learn matplotlib ripser persim
```

### Step 4: Verify Files

```bash
ls -lh
# Should see: conv_lstm.py, model.py, losses.py, train.py, run_train.sh, train_h200_lab.sub
```

### Step 5: Make Scripts Executable

```bash
chmod +x run_train.sh
mkdir -p logs
```

### Step 6: Submit Training Job

```bash
condor_submit train_h200_lab.sub
condor_q
```

### Step 7: Monitor Training

```bash
# Get job ID from condor_q output
# Then:
condor_tail <cluster.proc>
condor_tail -stderr <cluster.proc>

# Or check logs after completion:
tail -n 100 logs/train_<cluster>_<proc>.out
```

### Step 8: Check Results

```bash
# Results are saved to:
ls -lh /staging/groups/bhaskar_group/ivf/results/

# Or check local results:
ls -lh checkpoints/ logs/
```

## Troubleshooting

**Job stays Idle:**
```bash
condor_q -better-analyze <cluster.proc>
```

**Training fails:**
```bash
condor_tail -stderr <cluster.proc>
```

**Missing dataset_ivf:**
- The wrapper script automatically searches for it in parent directories
- If still fails, manually add path in `run_train.sh`





