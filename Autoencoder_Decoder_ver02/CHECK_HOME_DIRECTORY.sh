#!/bin/bash

du -sh ~

echo ""
du -sh ~/* ~/.[^.]* 2>/dev/null | sort -hr | head -20

echo ""
for dir in ~/.cache ~/.local ~/logs ~/tmp ~/.conda ~/.ipython; do
    if [ -d "$dir" ]; then
        echo "$dir:"
        du -sh "$dir"
    fi
done

echo ""
find ~ -type f -size +10M -exec du -sh {} \; 2>/dev/null | sort -hr | head -20

echo ""
find ~ -type f -name "*.log" -size +1M -exec du -sh {} \; 2>/dev/null | sort -hr | head -10

echo ""
quota -s






