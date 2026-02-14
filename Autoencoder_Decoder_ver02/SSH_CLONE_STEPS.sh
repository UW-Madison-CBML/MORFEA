#!/bin/bash
# Complete SSH setup and clone script for CHTC

echo "=== Setting up SSH and Cloning Repository ==="
echo ""

# Step 1: Check for existing SSH key
echo "1. Checking for SSH key..."
if [ -f ~/.ssh/id_ed25519.pub ]; then
    echo "✅ SSH key exists"
    cat ~/.ssh/id_ed25519.pub
    echo ""
    echo "If this key is already added to GitHub, skip to step 5"
else
    echo "❌ No SSH key found"
    echo ""
    echo "2. Generating SSH key..."
    ssh-keygen -t ed25519 -C "rho9@wisc.edu" -f ~/.ssh/id_ed25519 -N ""
    echo "✅ SSH key generated"
    echo ""
    echo "3. Your public key (copy this to GitHub):"
    cat ~/.ssh/id_ed25519.pub
    echo ""
    echo "4. Add this key to GitHub:"
    echo "   - Go to: https://github.com/settings/keys"
    echo "   - Click 'New SSH key'"
    echo "   - Paste the key above"
    echo "   - Click 'Add SSH key'"
    echo ""
    read -p "Press Enter after adding key to GitHub..."
fi

# Step 5: Test SSH connection
echo ""
echo "5. Testing SSH connection to GitHub..."
ssh -T git@github.com 2>&1

# Step 6: Clone repository
echo ""
echo "6. Cloning repository..."
cd ~
if [ -d "ivf" ]; then
    echo "Repository exists, updating..."
    cd ivf
    git pull
else
    git clone git@github.com:UW-Madison-CBML/ivf.git
    cd ivf
fi

# Step 7: Copy files
echo ""
echo "7. Copying files to ~/ivf_train..."
mkdir -p ~/ivf_train
if [ -d "Raffael/2025-11-19" ]; then
    cp Raffael/2025-11-19/train.py ~/ivf_train/
    cp Raffael/2025-11-19/model.py ~/ivf_train/
    cp Raffael/2025-11-19/conv_lstm.py ~/ivf_train/
    cp Raffael/2025-11-19/losses.py ~/ivf_train/
    echo "✅ Files copied"
else
    echo "❌ Raffael/2025-11-19 directory not found"
    echo "Available directories:"
    ls -la
fi

# Step 8: Verify files
echo ""
echo "8. Verifying files..."
cd ~/ivf_train
for file in train.py model.py conv_lstm.py losses.py; do
    if [ -f "$file" ]; then
        SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
        if [ "$SIZE" -lt 1000 ]; then
            echo "❌ $file: $SIZE bytes (TOO SMALL)"
        else
            echo "✅ $file: $SIZE bytes"
        fi
    else
        echo "❌ $file: NOT FOUND"
    fi
done

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  cd ~/ivf_train"
echo "  chmod +x run_train.sh"
echo "  mkdir -p logs"
echo "  condor_submit train_h200_lab.sub"
echo "  condor_q"





