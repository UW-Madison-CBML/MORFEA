#!/bin/bash
# CHTC Training Setup Script
# Clones from GitHub and sets up training environment

set -e

echo "=== CHTC ConvLSTM Autoencoder Training Setup ==="

# 1. Clone from GitHub
echo "1. Cloning repository from GitHub..."
cd ~
if [ -d "ivf" ]; then
    echo "   Repository exists, updating..."
    cd ivf
    git pull
else
    git clone https://github.com/UW-Madison-CBML/ivf.git
    cd ivf
fi

# 2. Navigate to model directory
echo "2. Navigating to model directory..."
cd Raffael/2025-11-19
pwd

# 3. Create Python virtual environment
echo "3. Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
echo "4. Installing dependencies..."
pip install --upgrade pip
pip install torch opencv-python pandas numpy tqdm scikit-learn matplotlib ripser persim

# 5. Test model can be imported
echo "5. Testing model import..."
python3 -c "from model import ConvLSTMAutoencoder; print('✓ Model import successful')"

# 6. Verify data connection (if index.csv exists)
if [ -f "../../index.csv" ]; then
    echo "6. Testing data connection..."
    python3 verify_data_connection.py || echo "   ⚠ Data connection test skipped (index.csv may need adjustment)"
else
    echo "6. index.csv not found in repo root, will use /project/bhaskar_group/ivf"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Ensure index.csv is accessible (or use shared data at /project/bhaskar_group/ivf)"
echo "2. Create/update train_h200_lab.sub"
echo "3. Submit: condor_submit train_h200_lab.sub"

