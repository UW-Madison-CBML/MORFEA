#!/bin/bash
# upload_to_staging_latents.sh

CHTC_USER="rho9"
CHTC_HOST="ap2001.chtc.wisc.edu"
STAGING_DIR="/staging/groups/bhaskar_group/rho9"

echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""

scp extract_all_latent_trajectories.py \
    extract_latents.sh \
    extract_latents.sub \
    model.py \
    dataset_ivf.py \
    build_index.py \
    index.csv \
    ${CHTC_USER}@${CHTC_HOST}:${STAGING_DIR}/

if [ $? -eq 0 ]; then
else
    exit 1
fi

echo ""

ssh ${CHTC_USER}@${CHTC_HOST} "mkdir -p ${STAGING_DIR}/checkpoints"

if [ -f "checkpoints/checkpoint_epoch_50.pt" ]; then
    scp checkpoints/checkpoint_epoch_50.pt \
        ${CHTC_USER}@${CHTC_HOST}:${STAGING_DIR}/checkpoints/
    
    if [ $? -eq 0 ]; then
    else
        exit 1
    fi
else
fi

echo ""
echo ""
echo ""
echo "   checkpoint = checkpoints/checkpoint_epoch_50.pt"
echo "   model_version = v1_baseline"
echo "   queue"
echo ""

