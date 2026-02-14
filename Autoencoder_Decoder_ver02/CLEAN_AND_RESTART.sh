#!/bin/bash

pkill -f "generate_tphate_for_aadhitya.py" 2>/dev/null
sleep 2

echo ""
rm -rf /staging/groups/bhaskar_group/rho9/aadhitya_v1_val
rm -rf /staging/groups/bhaskar_group/ivf/aadhitya_v1_val

echo ""
echo ""
echo "cd /staging/groups/bhaskar_group/rho9"
echo ""
echo "python3 generate_tphate_for_aadhitya.py \\"
echo "    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \\"
echo "    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \\"
echo "    --output_base /staging/groups/bhaskar_group/ivf/tphate_results \\"
echo "    --knn 5 \\"
echo "    > /dev/null 2>&1 &"






