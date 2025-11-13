#!/bin/bash
# detect_empty_wells.sh
pip uninstall opencv-python-headless
pip install opencv-python-headless
set -e

echo "Starting empty well detection and indexing..."

if [ -f "embryo_dataset.tar.gz" ]; then
    echo "Extracting embryo dataset..."
    tar -xzf embryo_dataset.tar.gz
fi

echo "Running empty well detection and building index..."
python build_index.py

if [ -f "index.csv" ]; then
    echo "Successfully created index.csv with empty_well labels"
    wc -l index.csv
else
    echo "Error: index.csv not created"
    exit 1
fi

