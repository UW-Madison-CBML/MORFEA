#!/bin/bash

cd /staging/groups/bhaskar_group/rho9

if [ -f "generate_tphate_for_aadhitya.py" ]; then
    mv generate_tphate_for_aadhitya.py generate_tphate_plots.py
else
fi

echo ""
if [ -d "aadhitya_v1_test" ]; then
    rm -rf aadhitya_v1_test
else
fi

echo ""
find . -iname "*aadhitya*" 2>/dev/null

echo ""
quota -s






