#!/bin/bash

echo ""

if [ -d "aadhitya_v1_val/tphate_plots" ]; then
    PROCESSED=$(ls -1 aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
    
    echo ""
    ls -t aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | head -5 | sed 's/.*\///; s/_tphate\.png//'
else
fi

echo ""
echo ""
echo "   condor_submit generate_tphate.sub"
echo ""
echo "   python3 generate_tphate_for_aadhitya.py \\"
echo "       --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \\"
echo "       --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \\"
echo "       --output_base aadhitya_v1_val \\"
echo "       --skip_existing"
echo ""






