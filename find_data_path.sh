#!/bin/bash
echo "=== Finding Data Path on GPU Node ==="
echo ""
echo "1. Checking /project/bhaskar_group/:"
ls -la /project/bhaskar_group/ 2>&1 | head -10
echo ""
echo "2. Checking /staging/groups/bhaskar_group/:"
ls -la /staging/groups/bhaskar_group/ 2>&1 | head -10
echo ""
echo "3. Searching for ivf directories:"
find /project -maxdepth 4 -type d -name "*ivf*" 2>/dev/null | head -10
find /staging -maxdepth 4 -type d -name "*ivf*" 2>/dev/null | head -10
