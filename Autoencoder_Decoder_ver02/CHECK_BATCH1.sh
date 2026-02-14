#!/bin/bash

cd /staging/groups/bhaskar_group/rho9

ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep
echo ""

if [ -f tphate_batch1.log ]; then
    echo ""
    tail -30 tphate_batch1.log
else
fi

echo ""
COUNT=$(ls -1 aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
echo ""

if [ $COUNT -gt 182 ]; then
    echo ""
    ls -t aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | head -5 | xargs -n1 basename | sed 's/_tphate\.png//'
else
fi






