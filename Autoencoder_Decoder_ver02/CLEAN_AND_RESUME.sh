#!/bin/bash

echo "============================================================"
echo "============================================================"
echo ""


cat << 'EOF'
OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/v1_baseline_tphate"
COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l || echo "0")
echo ""
df -h /staging/groups/bhaskar_group/rho9 | grep -E "(Filesystem|/dev/md9)"
echo ""

pkill -f "generate_tphate_plots.py"
sleep 2

if [ -d "/staging/groups/bhaskar_group/rho9/aadhitya_v1_val" ]; then
    OLD_SIZE=$(du -sh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val 2>/dev/null | cut -f1)
    echo ""
    echo "  rm -rf /staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
    echo ""
fi

du -sh /staging/groups/bhaskar_group/rho9/* 2>/dev/null | sort -h | tail -10
echo ""

echo ""

CURRENT_COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l || echo "0")
echo ""
echo "nohup python3 /staging/groups/bhaskar_group/rho9/generate_tphate_plots.py \\"
echo "    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \\"
echo "    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \\"
echo "    --output_base /staging/groups/bhaskar_group/rho9/v1_baseline_tphate \\"
echo "    --knn 5 \\"
echo "    --skip_existing \\"
echo "    > /tmp/tphate_run.log 2>&1 &"
echo ""
EOF

echo ""
echo "============================================================"





