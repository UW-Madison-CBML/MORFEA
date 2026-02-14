# 检查 rho9 目录详情

## 回到 rho9 目录检查

```bash
cd /staging/groups/bhaskar_group/rho9

# 1. 检查目录总大小
du -sh .

# 2. 所有目录大小
du -sh * 2>/dev/null | sort -hr

# 3. 所有大于 1M 的文件
find . -type f -size +1M -exec du -sh {} \; 2>/dev/null | sort -hr

# 4. 检查 v1_baseline_tphate 子目录
du -sh v1_baseline_tphate/* 2>/dev/null | sort -hr

# 5. 检查其他目录
ls -d */ 2>/dev/null
```

## 根据之前的信息

rho9 目录（677M）主要包含：
- v1_baseline_tphate (457M) - 当前输出，不能删
- checkpoints (151M) - 模型文件，不能删
- index.csv (20M) - 数据文件，不能删
- 其他小文件

**结论**：rho9 目录下可能没有太多可以删除的大文件。

## 选项

1. **继续运行，遇到配额错误时再处理**（可能只能处理部分胚胎）
2. **联系管理员增加配额**（如果需要处理所有 704 个胚胎）
3. **只处理 validation set**（减少处理数量）

先运行检查命令，看看 rho9 目录下还有什么可以清理的。






