# Setup SSH Key and Clone from GitHub

## Step 1: Check if SSH key exists on CHTC

```bash
ls -la ~/.ssh/id_*.pub
```

## Step 2: If no key exists, generate one

```bash
ssh-keygen -t ed25519 -C "rho9@wisc.edu"
# Press Enter to accept default location (~/.ssh/id_ed25519)
# Press Enter for no passphrase (or set one if you prefer)
```

## Step 3: Display public key

```bash
cat ~/.ssh/id_ed25519.pub
```

## Step 4: Add key to GitHub

1. Copy the output from step 3
2. Go to: https://github.com/settings/keys
3. Click "New SSH key"
4. Title: "CHTC-ap2001"
5. Paste the key
6. Click "Add SSH key"

## Step 5: Test SSH connection

```bash
ssh -T git@github.com
```

You should see: "Hi Grnho! You've successfully authenticated..."

## Step 6: Clone using SSH

```bash
cd ~
git clone git@github.com:UW-Madison-CBML/ivf.git
cd ivf/Raffael/2025-11-19
cp train.py model.py conv_lstm.py losses.py ~/ivf_train/
cd ~/ivf_train
ls -lh train.py model.py conv_lstm.py losses.py
```

## Step 7: Verify files and submit

```bash
# Verify file sizes (should be > 1000 bytes)
for file in train.py model.py conv_lstm.py losses.py; do
    SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
    echo "$file: $SIZE bytes"
done

# Submit
chmod +x run_train.sh
mkdir -p logs
condor_submit train_h200_lab.sub
condor_q
```



