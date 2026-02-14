# Direct Copy Solution - Files Too Large for GitHub Download

## Problem
GitHub download returns 404 or 14 bytes (corrupted). Files may not be accessible via raw URL.

## Solution: Copy File Contents Directly

Since files are too large to paste here, use one of these methods:

### Method 1: Use GitHub Web Interface

1. Go to: https://github.com/UW-Madison-CBML/ivf/tree/main/Raffael/2025-11-19
2. Click on each file (train.py, model.py, conv_lstm.py, losses.py)
3. Click "Raw" button
4. Copy entire content
5. On CHTC, create file and paste:
   ```bash
   nano train.py
   # Paste content, Ctrl+X, Y, Enter to save
   ```

### Method 2: Clone Repository (If SSH Works)

```bash
cd ~
git clone git@github.com:UW-Madison-CBML/ivf.git
cd ivf/Raffael/2025-11-19
cp train.py model.py conv_lstm.py losses.py ~/ivf_train/
```

### Method 3: Use Base64 Encoding (From Mac)

On your Mac:
```bash
cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"
base64 train.py | pbcopy
# Then paste into CHTC terminal:
# base64 -d > train.py
# Paste content, then Ctrl+D
```

### Method 4: Split Files into Smaller Chunks

Too complex, not recommended.

## Recommended: Use Method 1 (GitHub Web + Copy-Paste)

1. Open each file on GitHub
2. Click "Raw"
3. Select All (Cmd+A)
4. Copy (Cmd+C)
5. On CHTC: `nano train.py`, paste, save





