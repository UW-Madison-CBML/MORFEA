# Fix GitHub Authentication on CHTC

## Problem
`fatal: Authentication failed for 'https://github.com/UW-Madison-CBML/ivf.git/'`

## Solution Options

### Option 1: Use SSH Key (Recommended)

1. **Check if SSH key exists on CHTC:**
```bash
ls -la ~/.ssh/id_*.pub
```

2. **If no key exists, generate one:**
```bash
ssh-keygen -t ed25519 -C "rho9@wisc.edu"
# Press Enter to accept default location
# Press Enter for no passphrase (or set one)
```

3. **Display public key:**
```bash
cat ~/.ssh/id_ed25519.pub
```

4. **Add to GitHub:**
   - Copy the output
   - Go to: https://github.com/settings/keys
   - Click "New SSH key"
   - Paste and save

5. **Test SSH connection:**
```bash
ssh -T git@github.com
```

6. **Clone using SSH:**
```bash
git clone git@github.com:UW-Madison-CBML/ivf.git
```

### Option 2: Use Personal Access Token (PAT)

1. **Create PAT on GitHub:**
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token" → "Generate new token (classic)"
   - Name: "CHTC-ivf"
   - Select scope: `repo` (full control)
   - Generate and **copy the token**

2. **Clone using token:**
```bash
git clone https://YOUR_TOKEN@github.com/UW-Madison-CBML/ivf.git
```

Or when prompted for password, use the token.

### Option 3: Upload Files Directly (Fastest)

If authentication is too complicated, upload files directly from your Mac:

```bash
# On your Mac:
cd ~/ivf/Raffael/2025-11-19
scp -r * rho9@ap2001.chtc.wisc.edu:~/ivf_train/
```

Then on CHTC:
```bash
cd ~/ivf_train
chmod +x run_train.sh
mkdir -p logs
condor_submit train_h200_lab.sub
```





