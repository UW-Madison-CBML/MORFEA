#!/bin/bash
# Upload verification scripts to CHTC

echo "=== Uploading Verification Scripts to CHTC ==="
echo ""

CHTC_USER="rho9"
CHTC_HOST="ap2001.chtc.wisc.edu"
CHTC_DIR="~/ivf_repo"

echo "Uploading scripts..."
scp verify_dataset_size.sh verify_tar_contents.py ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Scripts uploaded successfully!"
    echo ""
    echo "Now SSH to CHTC and run:"
    echo "  ssh ${CHTC_USER}@${CHTC_HOST}"
    echo "  cd ~/ivf_repo"
    echo "  ./verify_dataset_size.sh"
    echo "  # OR"
    echo "  python3 verify_tar_contents.py"
else
    echo ""
    echo "❌ Upload failed. Check your SSH connection."
fi

