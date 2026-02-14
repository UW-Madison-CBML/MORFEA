#!/bin/bash

pkill -f "generate_tphate_plots.py"
sleep 2

echo ""
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
echo ""
echo "  T-PHATE plots: /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/"
echo "  Curvature plots: /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/curvature_plots/"
echo ""
du -sh /staging/groups/bhaskar_group/rho9/v1_baseline_tphate

echo ""
quota -s

echo ""

echo ""






