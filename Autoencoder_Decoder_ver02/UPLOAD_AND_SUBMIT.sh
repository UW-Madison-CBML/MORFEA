#!/bin/bash

set -e

echo ""

PROJECT_DIR="/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"
CHTC_HOST="rho9@ap2001.chtc.wisc.edu"
STAGING_DIR="/staging/groups/bhaskar_group/rho9"

cd "$PROJECT_DIR"


echo ""
echo ""

files=(
    "extract_all_latent_trajectories.py"
    "model.py"
    "dataset_ivf.py"
    "build_index.py"
    "extract_latents.sh"
)

for file in "${files[@]}"; do
done

echo ""
ssh "$CHTC_HOST" << 'REMOTE_EOF'
cd /staging/groups/bhaskar_group/rho9
for file in extract_all_latent_trajectories.py model.py dataset_ivf.py build_index.py extract_latents.sh; do
    if [ -f "$file" ]; then
        echo "    ✓ $file"
    else
        echo "    ✗ $file - MISSING!"
        exit 1
    fi
done

head -1 extract_latents.sh | grep -q "


REMOTE_EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
ssh "$CHTC_HOST" << 'REMOTE_EOF'
condor_q -format "%d " JobID -format "%s\n" JobStatus | grep "RUNNING" | awk '{print $1}' | while read jobid; do
    if [ -n "$jobid" ]; then
        condor_rm "$jobid" 2>/dev/null || true
    fi
done
REMOTE_EOF

echo ""
ssh "$CHTC_HOST" << 'REMOTE_EOF'
cd ~
if [ ! -f extract_latents_from_home.sub ]; then
    exit 1
fi

condor_submit extract_latents_from_home.sub
echo ""
echo ""
condor_q
REMOTE_EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo ""

