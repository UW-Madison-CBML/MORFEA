#!/bin/bash

cd /staging/groups/bhaskar_group/rho9

du -sh * 2>/dev/null | sort -hr

echo ""
find . -type f -size +10M -exec du -sh {} \; 2>/dev/null | sort -hr

echo ""
if [ -d "checkpoints" ]; then
    ls -lh checkpoints/
    echo ""
fi

echo ""
ls -d */ 2>/dev/null

echo ""
find . -type f \( -name "*.tar.gz" -o -name "*.zip" -o -name "*.tar" \) -exec du -sh {} \; 2>/dev/null | sort -hr

echo ""
quota -s






