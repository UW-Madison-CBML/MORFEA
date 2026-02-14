#!/bin/bash

echo "============================================================"
echo "============================================================"
echo ""
echo ""
echo "   cd \"/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02\""
echo "   scp extract_latents.sh rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/"
echo ""
echo "   grep -A 5 'Ensure DEVICE' /staging/groups/bhaskar_group/rho9/extract_latents.sh"
echo "
echo ""
echo "   grep -A 3 'if \[ -z \"\\\$DEVICE\" \]' /staging/groups/bhaskar_group/rho9/extract_latents.sh"
echo ""
echo "   condor_submit ~/extract_latents_from_home.sub"
echo ""
echo "============================================================"
echo "============================================================"








