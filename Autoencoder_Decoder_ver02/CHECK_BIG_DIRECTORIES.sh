#!/bin/bash

cd /staging/groups/bhaskar_group/rho9

du -sh * 2>/dev/null | sort -hr | head -10

echo ""
if [ -d "checkpoints" ]; then
    ls -lh checkpoints/ | head -20
    echo ""
fi

echo ""
find . -type f -size +10M -exec du -sh {} \; 2>/dev/null | sort -hr






