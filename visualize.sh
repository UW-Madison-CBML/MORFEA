#!/bin/bash

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <latents_csv> <cell_line>"
    echo "Example: $0 latents.csv \"cell_1,cell_2,cell_3\""
    exit 1
fi

LATENTS_CSV=$1
CELL_LINE=$2
OUTPUT_DIR="plots"

echo "Processing cell batch: $CELL_LINE"

# Check if plots.tar.gz exists - extract if present, create if not
if [ -f "plots.tar.gz" ]; then
    echo "Extracting existing plots..."
    tar -xzvf plots.tar.gz
else
    echo "Creating new plots directory..."
    mkdir -p "$OUTPUT_DIR"
fi

# Run visualization for this batch
python visualize.py "$LATENTS_CSV" "$CELL_LINE" --output "$OUTPUT_DIR"

# Rezip the plots
echo "Compressing plots..."
tar -czvf plots.tar.gz "$OUTPUT_DIR"
echo "Plots saved to plots.tar.gz"
