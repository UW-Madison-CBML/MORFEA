# 检查配额问题

## 检查命令

在 CHTC 上运行：

```bash
# 1. 检查配额
quota -s

# 2. 检查磁盘使用
df -h /staging/groups/bhaskar_group/rho9
df -h /staging/groups/bhaskar_group/ivf

# 3. 检查输出目录大小
du -sh /staging/groups/bhaskar_group/rho9/v1_baseline_tphate
du -sh /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots

# 4. 计算平均文件大小
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
du -sh /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots
echo "平均每个: $(($(du -sb /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots 2>/dev/null | awk '{print $1}') / COUNT / 1024 / 1024)) MB"

# 5. 检查 rho9 目录总大小
du -sh /staging/groups/bhaskar_group/rho9
```

## 可能的原因

1. **rho9 的 staging 目录有配额限制**（虽然通常不应该）
2. **文件太大**：即使 DPI 300，每个 plot 可能也有几 MB
3. **需要联系管理员**：如果确实有配额限制，可能需要请求增加配额

## 解决方案

如果确实有配额限制：
1. 进一步降低 DPI（从 150 到 100 或更低）
2. 使用 JPEG 格式而不是 PNG（文件更小）
3. 联系管理员增加配额
4. 或者保存到其他位置（如果有权限）

先运行检查命令，看看实际的配额和使用情况。






