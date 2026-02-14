#!/bin/bash

cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

echo ""
if [ -f "extract_all_latent_trajectories.py" ] && [ -f "extract_latents.sh" ]; then
    echo ""
    
    scp extract_all_latent_trajectories.py rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/
    
    echo ""
    scp extract_latents.sh rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/
    
    echo ""
    echo "============================================================"
    echo "============================================================"
else
    exit 1
fi








