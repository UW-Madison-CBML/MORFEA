# 修复配额问题

## 问题诊断
- 配额使用：39921M / 40960M (97.5%) - **接近上限！**
- 输出目录：456M（225 个胚胎）
- 平均每个胚胎：约 2MB（两种 plot）

## 解决方案

### 方案1: 降低 DPI 到 100（推荐）

已更新脚本，DPI 从 150 降低到 100，文件大小会减少约 2.25 倍。

```bash
# 1. 上传更新后的脚本
cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"
scp generate_tphate_for_aadhitya.py rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/

# 2. 在 CHTC 上继续运行
cd /staging/groups/bhaskar_group/rho9
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/rho9/v1_baseline_tphate \
    --knn 5 \
    --start_from $COUNT \
    > /tmp/tphate_run.log 2>&1 &
```

### 方案2: 清理一些空间

```bash
# 检查可以清理的文件
du -sh /staging/groups/bhaskar_group/rho9/* | sort -h

# 清理旧的日志文件
rm -f /staging/groups/bhaskar_group/rho9/tphate_*.log
rm -f /tmp/tphate_run.log
```

### 方案3: 分批处理并压缩

处理完一批后，压缩已完成的 plots，释放空间。

## 估计

- 当前：225 个胚胎 = 456M
- 剩余：479 个胚胎
- 如果 DPI 100：估计还需要约 400-500M
- **总估计：约 900M-1GB**

但配额只有 40960M，应该足够。问题可能是其他文件占用了空间。

先上传更新后的脚本（DPI 100），然后继续运行。






