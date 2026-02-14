#!/bin/bash

echo "============================================================"
echo "============================================================"
echo ""

echo "   finger jlundsgaard@wisc.edu"
echo ""

echo "   getent passwd jlundsgaard"
echo ""

echo ""
find /staging/groups/bhaskar_group/ivf/ -maxdepth 1 -name "README*" -o -name "*.txt" -o -name "*.md" 2>/dev/null | head -10

echo ""
ls -ld /staging/groups/bhaskar_group/ivf/ 2>/dev/null

echo ""
if [ -d "/staging/groups/bhaskar_group/ivf/.git" ]; then
    cd /staging/groups/bhaskar_group/ivf/
    git log --format='%an <%ae>' | sort -u | head -5 2>/dev/null
    cd - > /dev/null
else
fi

echo ""
echo "============================================================"
echo "============================================================"






