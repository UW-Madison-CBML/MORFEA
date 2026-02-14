#!/bin/bash
# Script to update train.py on CHTC with automatic index.csv check
# Run this on CHTC: bash UPDATE_TRAIN_ON_CHTC.sh

cd ~/ivf/Raffael/2025-11-19 || cd ~/ivf_train || { echo "ERROR: Cannot find working directory"; exit 1; }

echo "=== Updating train.py on CHTC ==="
echo "Current directory: $(pwd)"

# Backup original
cp train.py train.py.backup.$(date +%Y%m%d_%H%M%S)
echo "✓ Backed up original train.py"

# Check if the update is already there
if grep -q "★ Ensure index.csv exists before loading dataset ★" train.py; then
    echo "✓ train.py already has the update!"
    exit 0
fi

# Find the line number where we need to insert the check
# Look for the line: "# Dataset" or "train_dataset = IVFSequenceDataset"
INSERT_LINE=$(grep -n "train_dataset = IVFSequenceDataset" train.py | head -1 | cut -d: -f1)
if [ -z "$INSERT_LINE" ]; then
    INSERT_LINE=$(grep -n "# Dataset" train.py | head -1 | cut -d: -f1)
fi

if [ -z "$INSERT_LINE" ]; then
    echo "ERROR: Cannot find insertion point in train.py"
    exit 1
fi

echo "Found insertion point at line $INSERT_LINE"

# Create a temporary Python script to do the insertion
python3 << 'PYTHON_SCRIPT'
import sys

# Read the file
with open('train.py', 'r') as f:
    lines = f.readlines()

# Find insertion point (line before "train_dataset = IVFSequenceDataset")
insert_idx = None
for i, line in enumerate(lines):
    if 'train_dataset = IVFSequenceDataset' in line:
        insert_idx = i
        break
    if '# Dataset' in line and insert_idx is None:
        # Look for the next non-empty, non-comment line
        for j in range(i+1, len(lines)):
            if lines[j].strip() and not lines[j].strip().startswith('#'):
                insert_idx = j
                break
        if insert_idx:
            break

if insert_idx is None:
    print("ERROR: Cannot find insertion point")
    sys.exit(1)

# Check if already updated
if any('★ Ensure index.csv exists' in line for line in lines):
    print("✓ Already updated!")
    sys.exit(0)

# Insert the check code
check_code = """    # ★ Ensure index.csv exists before loading dataset ★
    index_path = Path(index_csv)
    if not index_path.exists():
        print(f"[train] {index_csv} not found, building with build_index.py ...", flush=True)
        if build_index is None:
            raise ImportError(
                f"[train] build_index module not available. "
                "Cannot auto-generate index.csv. Please run build_index.py manually."
            )
        build_index.main()
        if not index_path.exists():
            raise FileNotFoundError(
                f"[train] After running build_index.main(), still no {index_csv}. "
                "Check that symlink 'data' -> /project/bhaskar_group/ivf has valid content."
            )
        print(f"[train] ✓ Successfully created {index_csv}", flush=True)
    
"""

# Insert before the dataset loading line
lines.insert(insert_idx, check_code)

# Write back
with open('train.py', 'w') as f:
    f.writelines(lines)

print(f"✓ Successfully updated train.py at line {insert_idx}")
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo "✓ train.py updated successfully!"
    echo ""
    echo "Verifying update..."
    if grep -q "★ Ensure index.csv exists before loading dataset ★" train.py; then
        echo "✓ Verification passed!"
        echo ""
        echo "Testing syntax..."
        python3 -m py_compile train.py && echo "✓ Syntax OK!" || echo "✗ Syntax error!"
    else
        echo "✗ Update verification failed!"
        echo "Restoring backup..."
        mv train.py.backup.* train.py 2>/dev/null
        exit 1
    fi
else
    echo "✗ Update failed! Restoring backup..."
    mv train.py.backup.* train.py 2>/dev/null
    exit 1
fi





