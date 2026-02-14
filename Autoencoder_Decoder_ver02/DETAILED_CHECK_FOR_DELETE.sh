#!/bin/bash

cd /staging/groups/bhaskar_group/rho9

du -sh * .[^.]* 2>/dev/null | sort -hr

echo ""
echo ""
du -sh v1_baseline_tphate/*
ls -lh v1_baseline_tphate/tphate_plots/ | head -5
ls -lh v1_baseline_tphate/curvature_plots/ | head -5

echo ""
ls -lh checkpoints/

echo ""
for dir in */; do
    if [ "$dir" != "v1_baseline_tphate/" ] && [ "$dir" != "checkpoints/" ]; then
        echo "  $dir:"
        ls -lh "$dir" 2>/dev/null | head -10
        echo ""
    fi
done

echo ""
find . -type f -size +1M -exec du -sh {} \; 2>/dev/null | sort -hr

echo ""
find . -type f \( -name "*.tar.gz" -o -name "*.zip" -o -name "*.tar" -o -name "*.gz" \) -exec du -sh {} \; 2>/dev/null | sort -hr

echo ""
find . -type f -name "*.log" -exec du -sh {} \; 2>/dev/null | sort -hr

echo ""
if [ -d "tmp" ]; then
    ls -lha tmp/ | head -20
fi

echo ""
quota -s






