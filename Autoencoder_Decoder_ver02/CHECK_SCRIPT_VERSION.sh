#!/bin/bash

cd /staging/groups/bhaskar_group/rho9

if grep -q "start_from" generate_tphate_for_aadhitya.py; then
else
    echo ""
    echo "  cd \"/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02\""
    echo "  scp generate_tphate_for_aadhitya.py rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/"
fi






