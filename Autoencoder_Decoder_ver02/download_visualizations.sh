#!/bin/bash

cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"
mkdir -p tphate_results_final

echo "=== Downloading Visualization Files ==="
echo ""

echo "Downloading PNG files..."
scp "rho9@ap2001.chtc.wisc.edu:~/ivf_repo/tphate_segments_direct/*.png" tphate_results_final/ 2>&1

echo ""
echo "Downloading JSON files..."
scp "rho9@ap2001.chtc.wisc.edu:~/ivf_repo/tphate_segments_direct/*.json" tphate_results_final/ 2>&1 || echo "No JSON files found (optional)"

echo ""
echo "=== Download Complete ==="
echo ""
echo "Files in tphate_results_final/:"
ls -lh tphate_results_final/ | tail -20

