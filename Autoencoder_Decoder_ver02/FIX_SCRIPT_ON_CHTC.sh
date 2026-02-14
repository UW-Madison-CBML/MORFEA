#!/bin/bash

echo "============================================================"
echo "============================================================"
echo ""
echo ""
echo ""
echo "   head -1 /staging/groups/bhaskar_group/rho9/extract_latents.sh"
echo ""
echo "
echo "   cp /staging/groups/bhaskar_group/rho9/extract_latents.sh /staging/groups/bhaskar_group/rho9/extract_latents.sh.backup"
echo ""
echo "   cd \"/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02\""
echo "   scp extract_latents.sh rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/"
echo ""
echo "   sed -i '1s/.*/#!/bin/bash/' /staging/groups/bhaskar_group/rho9/extract_latents.sh"
echo ""
echo "   head -5 /staging/groups/bhaskar_group/rho9/extract_latents.sh"
echo ""
echo "   condor_rm 2852953.0
echo "   condor_submit ~/extract_latents_from_home.sub"
echo ""
echo "============================================================"

