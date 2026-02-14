#!/bin/bash

cd /staging/groups/bhaskar_group/rho9

ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep
if [ $? -ne 0 ]; then
else
fi
echo ""

if [ -f tphate_batch1.log ]; then
    tail -50 tphate_batch1.log
else
fi
echo ""

COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
echo ""

ls -t /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | head -5 | xargs -n1 basename | sed 's/_tphate\.png//'
echo ""







