#!/bin/bash

cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

scp generate_tphate_for_aadhitya.py rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/

echo ""
echo ""
echo ""
echo "cd /staging/groups/bhaskar_group/rho9"
echo ""
echo "
echo "nohup python3 generate_tphate_for_aadhitya.py \\"
echo "    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \\"
echo "    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \\"
echo "    --output_base aadhitya_v1_val \\"
echo "    --knn 5 \\"
echo "    --start_from 182 \\"
echo "    --max_embryos 200 \\"
echo "    > tphate_batch1.log 2>&1 &"
echo ""
echo "
echo "tail -f tphate_batch1.log"






