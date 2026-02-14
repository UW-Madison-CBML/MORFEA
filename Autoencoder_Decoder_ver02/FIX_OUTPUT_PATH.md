# 修复输出路径问题

## 问题
输出保存到了错误的位置（可能在 home 目录，有配额限制），导致 "Disk quota exceeded" 错误。

## 解决方案

使用**绝对路径**保存到 staging 目录：

```bash
cd /staging/groups/bhaskar_group/rho9

# 停止当前进程
pkill -f "generate_tphate_for_aadhitya.py"

# 使用绝对路径输出到 staging
nohup python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/rho9/aadhitya_v1_val \
    --knn 5 \
    --start_from 182 \
    --max_embryos 200 \
    > tphate_batch1.log 2>&1 &

# 查看日志
tail -f tphate_batch1.log
```

**关键改变**：
- ❌ `--output_base aadhitya_v1_val` (相对路径，可能存到 home)
- ✅ `--output_base /staging/groups/bhaskar_group/rho9/aadhitya_v1_val` (绝对路径，存到 staging)

## 检查当前输出位置

```bash
# 查看当前有多少 plots
ls -1 aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l
ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l

# 查看磁盘使用
df -h ~
df -h /staging/groups/bhaskar_group/rho9
```






