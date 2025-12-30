#!/bin/bash
# build_index_embryo.sh
set -e

echo "Starting embryo index building..."

if [ -f "embryo_dataset.tar.gz" ]; then
    echo "Extracting embryo dataset..."
    tar -xzf embryo_dataset.tar.gz
fi

echo "Building embryo index..."
python build_index_embryo.py "$@"

if [ -f "index_embryo.csv" ]; then
    echo "Successfully created index_embryo.csv"
    wc -l index_embryo.csv
else
    echo "Error: index_embryo.csv not created"
    exit 1
fi
