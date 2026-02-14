#!/bin/bash

cd /staging/groups/bhaskar_group/rho9

du -sh .

echo ""
du -sh * 2>/dev/null | sort -hr

echo ""
find . -type f -size +1M -exec du -sh {} \; 2>/dev/null | sort -hr

echo ""
echo "v1_baseline_tphate:"
du -sh v1_baseline_tphate/* 2>/dev/null | sort -hr

echo ""
echo "checkpoints:"
ls -lh checkpoints/

echo ""
ls -d */ 2>/dev/null | grep -v "v1_baseline_tphate\|checkpoints"

echo ""
du -sh .[^.]* 2>/dev/null | sort -hr

echo ""
quota -s






