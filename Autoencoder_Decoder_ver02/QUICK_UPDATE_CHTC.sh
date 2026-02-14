#!/bin/bash

CHTC_USER="rho9"
CHTC_HOST="ap2001.chtc.wisc.edu"
CHTC_DIR="/staging/groups/bhaskar_group/rho9"

echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

scp extract_latents.sh \
    extract_all_latent_trajectories.py \
    ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/

if [ $? -eq 0 ]; then
    echo ""
    echo ""
else
    exit 1
fi

