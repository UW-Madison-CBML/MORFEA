# 测试 ivf 目录写入权限

## 在 CHTC 上运行

```bash
# 测试写入权限
TEST_FILE="/staging/groups/bhaskar_group/ivf/test_write_$(date +%s)"
touch "$TEST_FILE" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ 可以创建文件"
    rm "$TEST_FILE"
    
    # 测试创建目录
    TEST_DIR="/staging/groups/bhaskar_group/ivf/test_dir_$(date +%s)"
    mkdir -p "$TEST_DIR" 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ 可以创建目录"
        rmdir "$TEST_DIR"
        echo ""
        echo "结论：可以在 ivf 目录写入！"
    else
        echo "✗ 无法创建目录"
    fi
else
    echo "✗ 无法创建文件"
fi
```

如果可以在 ivf 目录写入，就可以把输出保存到那里，不受个人配额限制。






