#!/bin/bash

echo "============================================================"
echo "============================================================"
echo ""

OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/v1_baseline_tphate"
COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l || echo "0")
echo ""

df -h /staging/groups/bhaskar_group/rho9 | grep -E "(Filesystem|/dev/md9)"
echo ""

pkill -f "generate_tphate_plots.py" 2>/dev/null
pkill -f "generate_tphate_for_aadhitya.py" 2>/dev/null
sleep 2
echo ""

if [ -d "/staging/groups/bhaskar_group/rho9/aadhitya_v1_val" ]; then
    OLD_SIZE=$(du -sh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val 2>/dev/null | cut -f1)
    rm -rf /staging/groups/bhaskar_group/rho9/aadhitya_v1_val
    echo ""
fi

df -h /staging/groups/bhaskar_group/rho9 | grep -E "(Filesystem|/dev/md9)"
echo ""

echo ""
nohup python3 /staging/groups/bhaskar_group/rho9/generate_tphate_plots.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/rho9/v1_baseline_tphate \
    --knn 5 \
    --skip_existing \
    > /tmp/tphate_run.log 2>&1 &

PID=$!
echo ""
echo "  tail -f /tmp/tphate_run.log"
echo ""
echo "  while true; do COUNT=\$(ls -1 $OUTPUT_DIR/tphate_plots/*.png 2>/dev/null | wc -l); echo \"\$(date '+%H:%M:%S'): \$COUNT / 704\"; sleep 30; done"
echo ""
echo "============================================================"






