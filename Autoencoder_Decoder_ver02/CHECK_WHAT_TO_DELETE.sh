#!/bin/bash

quota -s

echo ""
cd /staging/groups/bhaskar_group/rho9
du -sh * 2>/dev/null | sort -h

echo ""
find . -name "*.log" -type f -exec du -sh {} \; 2>/dev/null | sort -h

echo ""
du -sh tmp 2>/dev/null
ls -lh tmp/ 2>/dev/null | head -20

echo ""
ls -d */ 2>/dev/null | while read dir; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
        echo "$size  $dir"
    fi
done | sort -h

echo ""
du -sh ~/logs 2>/dev/null
ls -lh ~/logs/*.log 2>/dev/null | head -20






