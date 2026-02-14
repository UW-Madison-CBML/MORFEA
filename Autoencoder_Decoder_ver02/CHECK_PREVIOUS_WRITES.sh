#!/bin/bash

find /staging/groups/bhaskar_group/ivf/ -user rho9 -ls 2>/dev/null | head -20

echo ""
ls -ld /staging/groups/bhaskar_group/ivf/plots/
ls -la /staging/groups/bhaskar_group/ivf/plots/ 2>/dev/null | head -20

echo ""
TEST_FILE="/staging/groups/bhaskar_group/ivf/plots/test_$(date +%s)"
touch "$TEST_FILE" 2>&1
if [ $? -eq 0 ]; then
    rm "$TEST_FILE"
else
fi

echo ""
python3 -c "
import os
test_dir = '/staging/groups/bhaskar_group/ivf/test_python_dir'
try:
    os.makedirs(test_dir, exist_ok=True)
    print('✓ Python 可以创建目录')
    os.rmdir(test_dir)
except Exception as e:
    print(f'✗ Python 无法创建目录: {e}')
"






