#!/bin/bash

cd /staging/groups/bhaskar_group/rho9


rm -rf ~/.cache

if [ -f ~/ivf-embryo-analysis-Raffael.tgz ]; then
    rm ~/ivf-embryo-analysis-Raffael.tgz
fi

echo ""
quota -s

echo ""
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)

echo ""
python3 generate_tphate_plots.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/rho9/v1_baseline_tphate \
    --knn 5 \
    --start_from $COUNT \
    > /tmp/tphate_run.log 2>&1 &

echo ""
echo "  while true; do"
echo "    COUNT=\$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)"
echo "    echo \"[\$(date '+%H:%M:%S')] \$COUNT / 704 plots\""
echo "    sleep 30"
echo "  done"






