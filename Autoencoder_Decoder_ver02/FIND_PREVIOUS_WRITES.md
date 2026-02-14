# 检查之前是否有写入记录

## 可能的情况

1. **之前在某个子目录写入**（如 plots 目录）
2. **Python 脚本可能有不同权限**
3. **之前权限不同，后来被更改了**

## 检查命令

```bash
# 1. 检查是否有 rho9 创建的文件
find /staging/groups/bhaskar_group/ivf/ -user rho9 -ls 2>/dev/null

# 2. 检查 plots 目录
ls -ld /staging/groups/bhaskar_group/ivf/plots/
ls -la /staging/groups/bhaskar_group/ivf/plots/

# 3. 尝试在 plots 目录写入
touch /staging/groups/bhaskar_group/ivf/plots/test_$(date +%s)
# 如果可以，删除测试文件

# 4. 或者尝试用 Python 创建（可能权限不同）
python3 -c "import os; os.makedirs('/staging/groups/bhaskar_group/ivf/test_python', exist_ok=True)"
```

如果找到之前可以写入的子目录，可以在那里保存输出。






