#!/bin/bash

REMOTE_USER="rho9"
REMOTE_HOST="ap2001.chtc.wisc.edu"
REMOTE_DIR="~/ivf_repo/tphate_segments_direct"
LOCAL_DIR="$HOME/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02/tphate_results/tphate_segments_direct"

mkdir -p "$LOCAL_DIR"

echo "=== Downloading Remaining Files ==="
echo ""

files=(
    "segment_B_frames.png"
    "segment_C_frames.png"
    "segment_D_frames.png"
    "tphate_segments_combined.png"
    "segments_metadata.json"
)

for file in "${files[@]}"; do
    if [ -f "$LOCAL_DIR/$file" ]; then
        echo "✓ $file (already exists)"
        continue
    fi
    
    echo ""
    echo "Downloading $file..."
    echo "  (If timeout, wait a moment and try again manually)"
    
    if scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${file} "$LOCAL_DIR/"; then
        echo "  ✓ Success!"
    else
        echo "  ✗ Failed. You can retry manually with:"
        echo "    scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${file} $LOCAL_DIR/"
    fi
    
    sleep 2
done

echo ""
echo "=========================================="
echo "Final Status:"
echo "=========================================="
ls -lh "$LOCAL_DIR" 2>/dev/null

