# 清理并重命名（去掉人名）

## 查找包含 aadhitya 的文件/目录

```bash
cd /staging/groups/bhaskar_group/rho9

# 查找所有包含 aadhitya 的文件和目录
find . -iname "*aadhitya*" 2>/dev/null

# 检查目录
ls -d */ 2>/dev/null | grep -i aadhitya
```

## 清理和重命名

### 1. 删除测试目录
```bash
# 删除 aadhitya_v1_test（如果存在）
rm -rf aadhitya_v1_test
```

### 2. 检查当前输出目录

当前输出目录应该是 `v1_baseline_tphate`，这个名字没有问题（没有包含人名）。

如果还有其他包含人名的目录，可以：
- 删除（如果是旧的/不需要的）
- 或者重命名

## 检查脚本文件

```bash
# 检查脚本中是否有人名
grep -i "aadhitya" *.py *.sh 2>/dev/null
```

脚本文件中的注释或文档可以有人名，但文件和目录名不应该有人名。

运行查找命令，看看还有什么需要清理的。






