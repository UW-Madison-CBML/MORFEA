#!/bin/bash

du -sh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val 2>/dev/null

echo ""
ls -ld /staging/groups/bhaskar_group/ivf 2>/dev/null
du -sh /staging/groups/bhaskar_group/ivf 2>/dev/null

echo ""
quota -s 2>/dev/null || df -h /staging/groups/bhaskar_group/rho9

echo ""






