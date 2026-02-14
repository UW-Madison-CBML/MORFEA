#!/bin/bash
# 测试 ivf 目录的写入权限

echo "=== 检查 ivf 目录权限 ==="
ls -ld /staging/groups/bhaskar_group/ivf/

echo ""
echo "=== 测试写入权限 ==="
# 尝试创建测试文件
TEST_FILE="/staging/groups/bhaskar_group/ivf/test_write_permission_$(date +%s)"
if touch "$TEST_FILE" 2>/dev/null; then
    echo "✓ 可以在 ivf 目录创建文件"
    rm "$TEST_FILE"
    
    echo ""
    echo "=== 尝试创建目录 ==="
    TEST_DIR="/staging/groups/bhaskar_group/ivf/test_dir_$(date +%s)"
    if mkdir -p "$TEST_DIR" 2>/dev/null; then
        echo "✓ 可以在 ivf 目录创建目录"
        rmdir "$TEST_DIR"
        
        echo ""
        echo "=== 结论 ==="
        echo "可以在 ivf 目录写入！可以保存输出到 ivf 目录，不受个人配额限制。"
    else
        echo "✗ 无法在 ivf 目录创建目录"
        echo "错误信息："
        mkdir -p "$TEST_DIR" 2>&1
    fi
else
    echo "✗ 无法在 ivf 目录创建文件"
    echo "错误信息："
    touch "$TEST_FILE" 2>&1
fi






