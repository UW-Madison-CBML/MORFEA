#!/bin/bash

pkill -f "generate_tphate_for_aadhitya.py" 2>/dev/null
sleep 2

echo ""

if [ -d "/staging/groups/bhaskar_group/rho9/aadhitya_v1_val" ]; then
    rm -rf /staging/groups/bhaskar_group/rho9/aadhitya_v1_val
fi

if [ -d "/staging/groups/bhaskar_group/ivf/aadhitya_v1_val" ]; then
    rm -rf /staging/groups/bhaskar_group/ivf/aadhitya_v1_val
fi

echo ""






