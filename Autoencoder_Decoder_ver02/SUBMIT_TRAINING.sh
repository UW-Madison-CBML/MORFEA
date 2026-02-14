#!/bin/bash
# Complete training submission script for CHTC
# Run this on CHTC access point

set -e

echo "=== ConvLSTM Autoencoder Training Submission ==="
echo ""

# Check if we're in the right directory
if [ ! -f "train_h200_lab.sub" ]; then
    echo "Error: train_h200_lab.sub not found!"
    echo "Please run this script from the directory containing train_h200_lab.sub"
    exit 1
fi

# 1. Verify all required files exist
echo "1. Verifying required files..."
REQUIRED_FILES=("train.py" "model.py" "losses.py" "run_train.sh" "train_h200_lab.sub" "dataset_ivf.py" "build_index.py")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo "❌ Missing files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "Download missing files from GitHub:"
    echo "  curl -L -o <file> \"https://raw.githubusercontent.com/UW-Madison-CBML/ivf/main/Raffael/2025-11-19/<file>\""
    exit 1
fi

echo "✅ All required files present"
echo ""

# 2. Make scripts executable
echo "2. Setting permissions..."
chmod +x run_train.sh
mkdir -p logs
echo "✅ Permissions set"
echo ""

# 3. Display submit file configuration
echo "3. Submit file configuration:"
echo "   - GPU: H200 (1 GPU)"
echo "   - Memory: 16GB"
echo "   - Disk: 30GB"
echo "   - Project: UWMadison_BME_Bhaskar"
echo ""

# 4. Submit job
echo "4. Submitting training job..."
condor_submit train_h200_lab.sub

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Job submitted successfully!"
    echo ""
    
    # 5. Show job status
    echo "5. Current job status:"
    sleep 2
    condor_q
    echo ""
    
    # Get job ID
    JOB_ID=$(condor_q -format "%s.%s\n" ClusterId ProcId | head -1)
    if [ -n "$JOB_ID" ]; then
        echo "Job ID: $JOB_ID"
        echo ""
        echo "=== Next Steps ==="
        echo ""
        echo "Monitor training:"
        echo "  condor_tail $JOB_ID"
        echo ""
        echo "Check status:"
        echo "  condor_q"
        echo ""
        echo "View errors:"
        echo "  condor_tail -stderr $JOB_ID"
        echo ""
        echo "Cancel job (if needed):"
        echo "  condor_rm $JOB_ID"
    fi
else
    echo "❌ Job submission failed!"
    exit 1
fi


