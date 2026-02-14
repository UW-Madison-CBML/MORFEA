#!/bin/bash

ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep
if [ $? -ne 0 ]; then
else
fi

echo ""

echo ""
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)

echo ""
ls -t /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | head -5 | xargs -n1 basename | sed 's/_tphate\.png//'






