#!/bin/bash

cd /staging/groups/bhaskar_group/rho9

if ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep > /dev/null; then
    ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep
else
fi
echo ""

if [ -f tphate_batch1.log ]; then
    tail -30 tphate_batch1.log
else
fi
echo ""

COUNT=$(ls -1 aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
EXPECTED_AFTER_BATCH1=382
REMAINING=$((EXPECTED_AFTER_BATCH1 - COUNT))

if [ $COUNT -gt 182 ]; then
    echo ""
    ls -t aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | head -3 | xargs -n1 basename | sed 's/_tphate\.png//'
elif [ $COUNT -eq 182 ]; then
fi

echo ""






