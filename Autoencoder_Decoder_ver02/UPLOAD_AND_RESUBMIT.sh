#!/bin/bash

REMOTE_USER="rho9"
REMOTE_HOST="ap2001.chtc.wisc.edu"
STAGING_DIR="/staging/groups/bhaskar_group/rho9"

echo "============================================================"
echo "============================================================"
echo ""

cd "$(dirname "$0")"

if [ ! -f "extract_latents.sh" ]; then
    exit 1
fi

if [ ! -f "extract_latents_from_home.sub" ]; then
    exit 1
fi

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
echo ""
echo "   ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo ""
echo "   condor_rm 2851275.0"
echo ""
echo "   tail -10 ~/extract_latents_from_home.sub"
echo ""
echo "   checkpoint = /staging/groups/bhaskar_group/rho9/checkpoints/checkpoint_epoch_50.pt"
echo "   model_version = v1_baseline"
echo "   queue"
echo ""
echo "   echo 'checkpoint = /staging/groups/bhaskar_group/rho9/checkpoints/checkpoint_epoch_50.pt' >> ~/extract_latents_from_home.sub"
echo "   echo 'model_version = v1_baseline' >> ~/extract_latents_from_home.sub"
echo "   echo 'queue' >> ~/extract_latents_from_home.sub"
echo "   condor_submit ~/extract_latents_from_home.sub"
echo ""
echo "   condor_q -submitter rho9"
echo ""
echo "   condor_tail -f <job_id>"
echo ""
echo "============================================================"

