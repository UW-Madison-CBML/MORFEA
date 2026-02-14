#!/bin/bash

echo "============================================================"
echo "============================================================"
echo ""
echo ""
cat << 'EOF'
OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/v1_baseline_tphate"
T_PHATE_COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l || echo "0")
CURVATURE_COUNT=$(ls -1 "$OUTPUT_DIR/curvature_plots"/*.png 2>/dev/null | wc -l || echo "0")

echo "T-PHATE plots: $T_PHATE_COUNT / 704"
echo "Curvature plots: $CURVATURE_COUNT / 704"
echo ""

df -h /staging/groups/bhaskar_group/rho9 | grep -E "(Filesystem|/dev/md9)"
echo ""

if ps aux | grep "generate_tphate" | grep -v grep > /dev/null; then
    ps aux | grep "generate_tphate" | grep -v grep | head -1
else
fi
echo ""

ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -5 | xargs -n1 basename | sed 's/_tphate.png//'
EOF

echo ""
echo "============================================================"
