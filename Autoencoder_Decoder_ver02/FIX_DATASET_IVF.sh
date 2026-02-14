#!/bin/bash
# Fix missing dataset_ivf.py issue

cd ~/ivf_train

echo "=== Fixing dataset_ivf.py Issue ==="
echo ""

# 1. Check if repo has the file
echo "1. Checking repository for dataset_ivf.py..."
if [ -f ~/ivf/Code/dataset_ivf.py ]; then
    echo "✅ Found in ~/ivf/Code/dataset_ivf.py"
    cp ~/ivf/Code/dataset_ivf.py .
    echo "✅ Copied to current directory"
elif [ -f ~/ivf/dataset_ivf.py ]; then
    echo "✅ Found in ~/ivf/dataset_ivf.py"
    cp ~/ivf/dataset_ivf.py .
    echo "✅ Copied to current directory"
else
    echo "❌ Not found in repo, searching..."
    find ~/ivf -name "dataset_ivf.py" 2>/dev/null | head -1
fi
echo ""

# 2. Also check for build_index.py
echo "2. Checking for build_index.py..."
if [ -f ~/ivf/Code/build_index.py ]; then
    echo "✅ Found build_index.py"
    cp ~/ivf/Code/build_index.py .
elif [ -f ~/ivf/build_index.py ]; then
    echo "✅ Found build_index.py"
    cp ~/ivf/build_index.py .
else
    echo "⚠️  build_index.py not found (may not be needed)"
fi
echo ""

# 3. Update run_train.sh to use local files
echo "3. Updating run_train.sh..."
cat > run_train.sh << 'EOF'
#!/bin/bash
# Training wrapper script for CHTC

# Use shared dataset
ln -sfn /project/bhaskar_group/ivf data

# Install dependencies
PYDEPS="$PWD/pydeps"
mkdir -p "$PYDEPS"
python3 -m pip install --no-cache-dir --upgrade pip
python3 -m pip install --no-cache-dir opencv-python pandas numpy tqdm scikit-learn matplotlib ripser persim -t "$PYDEPS"
export PYTHONPATH="$PYDEPS:$PYTHONPATH"

# Add current directory to path (for dataset_ivf.py)
export PYTHONPATH="$PWD:$PYTHONPATH"

# Build index if needed
if [ ! -f index.csv ]; then
    if [ -f "build_index.py" ]; then
        python3 build_index.py || echo "Index build failed, continuing..."
    elif [ -f "../../Code/build_index.py" ]; then
        python3 ../../Code/build_index.py || echo "Index build failed, continuing..."
    fi
fi

# Run training (don't exit on error)
echo "Starting training..."
python3 train.py \
    --index_csv index.csv \
    --batch_size 8 \
    --seq_len 20 \
    --num_epochs 50 \
    --learning_rate 3e-4 \
    --save_dir checkpoints \
    --log_dir logs || echo "Training failed, but continuing to package results..."

# ALWAYS create results.tgz (even if training failed)
echo "Packaging results..."
if [ -d "checkpoints" ] && [ -d "logs" ]; then
    tar -czf results.tgz checkpoints/ logs/ 2>/dev/null || tar -czf results.tgz logs/
elif [ -d "logs" ]; then
    tar -czf results.tgz logs/
elif [ -d "checkpoints" ]; then
    tar -czf results.tgz checkpoints/
else
    echo "empty" > empty.txt && tar -czf results.tgz empty.txt && rm -f empty.txt
fi

# Verify results.tgz exists
if [ ! -f results.tgz ]; then
    echo "empty" > empty.txt && tar -czf results.tgz empty.txt && rm -f empty.txt
fi

echo "Results packaged: $(ls -lh results.tgz)"
EOF

chmod +x run_train.sh
echo "✅ run_train.sh updated"
echo ""

# 4. Update submit file to include dataset_ivf.py
echo "4. Updating submit file to include dataset_ivf.py..."
if [ -f "dataset_ivf.py" ]; then
    # Update transfer_input_files
    sed -i 's/transfer_input_files = run_train.sh, train.py, model.py, conv_lstm.py, losses.py/transfer_input_files = run_train.sh, train.py, model.py, conv_lstm.py, losses.py, dataset_ivf.py/' train_h200_lab.sub
    echo "✅ Submit file updated to include dataset_ivf.py"
else
    echo "⚠️  dataset_ivf.py not found, submit file not updated"
fi
echo ""

# 5. Verify files
echo "5. Verifying files:"
ls -lh dataset_ivf.py build_index.py run_train.sh 2>/dev/null | grep -v "cannot access"
echo ""

echo "=== Ready to Resubmit ==="
echo "Next: condor_rm 2650361.0 && condor_submit train_h200_lab.sub"





