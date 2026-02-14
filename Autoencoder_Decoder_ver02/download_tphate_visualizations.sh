#!/bin/bash

REMOTE_USER="rho9"
REMOTE_HOST="ap2001.chtc.wisc.edu"
REMOTE_DIR="~/ivf_repo"
LOCAL_DIR="$HOME/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02/tphate_results"

mkdir -p "$LOCAL_DIR/tphate_segments_direct"

echo "=== Downloading TPHATE Visualizations ==="
echo ""

echo "Checking remote files..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "ls -lh ${REMOTE_DIR}/tphate_segments_direct/" 2>/dev/null

echo ""
echo "Downloading files..."

scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/tphate_segments_direct/*.png "$LOCAL_DIR/tphate_segments_direct/" 2>/dev/null && echo "✓ PNG files downloaded" || echo "⚠️  PNG files not found"

scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/tphate_segments_direct/*.json "$LOCAL_DIR/tphate_segments_direct/" 2>/dev/null && echo "✓ JSON files downloaded" || echo "⚠️  JSON files not found"

echo ""
echo "Local files:"
ls -lh "$LOCAL_DIR/tphate_segments_direct/" 2>/dev/null || echo "  (directory is empty)"
