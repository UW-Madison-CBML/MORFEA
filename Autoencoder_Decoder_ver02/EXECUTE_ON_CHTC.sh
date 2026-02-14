#!/bin/bash
# Copy and paste this entire script into CHTC terminal
# Or run: bash <(curl -s https://raw.githubusercontent.com/.../EXECUTE_ON_CHTC.sh)

set -e

echo "=== CHTC ConvLSTM Autoencoder Training Setup ==="
echo ""

# Step 1: Clone repository
echo "Step 1: Cloning repository..."
cd ~
if [ -d "ivf" ]; then
    echo "Repository exists, updating..."
    cd ivf
    git pull
    cd Raffael/2025-11-19
else
    git clone https://github.com/UW-Madison-CBML/ivf.git
    cd ivf/Raffael/2025-11-19
fi

echo "✓ Repository ready"
echo ""

# Step 2: Verify files
echo "Step 2: Verifying files..."
if [ ! -f "train.py" ] || [ ! -f "model.py" ] || [ ! -f "run_train.sh" ] || [ ! -f "train_h200_lab.sub" ]; then
    echo "Error: Required files not found!"
    ls -lh
    exit 1
fi
echo "✓ All files present"
echo ""

# Step 3: Make scripts executable
echo "Step 3: Setting permissions..."
chmod +x run_train.sh
mkdir -p logs
echo "✓ Permissions set"
echo ""

# Step 4: Submit job
echo "Step 4: Submitting training job..."
condor_submit train_h200_lab.sub

echo ""
echo "=== Job Submitted ==="
echo ""
echo "Monitor with:"
echo "  condor_q"
echo "  condor_tail <cluster.proc>"
echo ""
echo "Check results at:"
echo "  /staging/groups/bhaskar_group/ivf/results/"
echo ""





