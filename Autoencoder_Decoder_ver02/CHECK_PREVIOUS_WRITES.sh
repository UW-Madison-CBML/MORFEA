#!/bin/bash
# 检查之前是否有在 ivf 目录写入的记录

echo "=== 检查 ivf 目录下是否有 rho9 创建的文件/目录 ==="
find /staging/groups/bhaskar_group/ivf/ -user rho9 -ls 2>/dev/null | head -20

echo ""
echo "=== 检查 plots 目录（之前可能在这里写入过）==="
ls -ld /staging/groups/bhaskar_group/ivf/plots/
ls -la /staging/groups/bhaskar_group/ivf/plots/ 2>/dev/null | head -20

echo ""
echo "=== 检查是否可以写入到 plots 目录 ==="
TEST_FILE="/staging/groups/bhaskar_group/ivf/plots/test_$(date +%s)"
touch "$TEST_FILE" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ 可以在 plots 子目录写入！"
    rm "$TEST_FILE"
else
    echo "✗ 无法在 plots 子目录写入"
fi

echo ""
echo "=== 或者检查是否可以通过 Python 脚本创建（可能权限不同）==="
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






