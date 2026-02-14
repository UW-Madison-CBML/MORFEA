# GitHub Upload Guide

## Authentication Setup

GitHub no longer supports password authentication. You need either:

### Option 1: Use SSH Key (Recommended)

1. **Check if you have SSH key**:
   ```bash
   ls -la ~/.ssh/id_*.pub
   ```

2. **If no key exists, create one**:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # Press Enter to accept default location
   # Press Enter for no passphrase (or set one)
   ```

3. **Add SSH key to GitHub**:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
   - Copy the output
   - Go to: https://github.com/settings/keys
   - Click "New SSH key"
   - Paste the key and save

4. **Test SSH connection**:
   ```bash
   ssh -T git@github.com
   ```

5. **Clone using SSH**:
   ```bash
   git clone git@github.com:UW-Madison-CBML/ivf.git
   ```

### Option 2: Use Personal Access Token (PAT)

1. **Create PAT**:
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token" → "Generate new token (classic)"
   - Name: "ivf-upload"
   - Select scopes: `repo` (full control)
   - Generate and **copy the token** (you won't see it again!)

2. **Clone using PAT**:
   ```bash
   git clone https://YOUR_TOKEN@github.com/UW-Madison-CBML/ivf.git
   ```
   Or use token as password when prompted.

## Upload Files

Once authenticated, follow these steps:

```bash
# 1. Clone repository (if not already)
git clone git@github.com:UW-Madison-CBML/ivf.git
cd ivf

# 2. Create folder structure
mkdir -p Raffael/2025-11-19

# 3. Copy files from your local directory
cp -r "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"/* Raffael/2025-11-19/

# 4. Remove unnecessary files
cd Raffael/2025-11-19
rm -f QUALITY_ASSESSMENT.md DATA_CONNECTION_VERIFIED.md GITHUB_SETUP.md upload_to_github.sh GITHUB_UPLOAD_GUIDE.md

# 5. Go back to repo root
cd ../..

# 6. Check what will be added
git status

# 7. Add files
git add Raffael/

# 8. Commit
git commit -m "Add ConvLSTM Autoencoder implementation (Raffael/2025-11-19)"

# 9. Push
git push
```

## Quick One-Liner (After Authentication Setup)

```bash
cd ~ && \
git clone git@github.com:UW-Madison-CBML/ivf.git && \
cd ivf && \
mkdir -p Raffael/2025-11-19 && \
cp -r "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"/* Raffael/2025-11-19/ && \
cd Raffael/2025-11-19 && \
rm -f QUALITY_ASSESSMENT.md DATA_CONNECTION_VERIFIED.md GITHUB_SETUP.md upload_to_github.sh GITHUB_UPLOAD_GUIDE.md && \
cd ../.. && \
git add Raffael/ && \
git commit -m "Add ConvLSTM Autoencoder (Raffael/2025-11-19)" && \
git push
```





