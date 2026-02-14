#!/bin/bash

REMOTE_USER="rho9"
REMOTE_HOST="ap2001.chtc.wisc.edu"
STAGING_DIR="/staging/groups/bhaskar_group/rho9"

echo "============================================================"
echo "============================================================"
echo ""

cd "$(dirname "$0")"
echo ""

if [ ! -f "extract_latents.sh" ]; then
    exit 1
fi

if [ ! -f "extract_latents_from_home.sub" ]; then
    exit 1
fi

ls -lh extract_latents.sh extract_latents_from_home.sub
echo ""

scp extract_latents.sh ${REMOTE_USER}@${REMOTE_HOST}:${STAGING_DIR}/

if [ $? -eq 0 ]; then
else
    exit 1
fi

echo ""
scp extract_latents_from_home.sub ${REMOTE_USER}@${REMOTE_HOST}:~/

if [ $? -eq 0 ]; then
else
    exit 1
fi

echo ""
echo "============================================================"
echo "============================================================"
echo ""
echo ""
echo "   ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo ""
echo ""
echo "============================================================"

