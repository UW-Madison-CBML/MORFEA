# 降低 DPI 后继续运行

## 问题
- CPU 时间限制导致进程被终止
- 磁盘配额超限（即使使用 staging 目录）

## 解决方案
已将 DPI 从 300 降低到 150，文件大小会减少约 4 倍。

## 步骤

### 1. 检查当前进度

```bash
# 查看已完成的数量
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
echo "已完成: $COUNT 个 plots"
```

### 2. 上传更新后的脚本（降低 DPI）

```bash
# 在本地运行
cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"
scp generate_tphate_for_aadhitya.py rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/
```

### 3. 从当前进度继续运行

假设已完成 225 个，从第 226 个开始继续处理 200 个：

```bash
cd /staging/groups/bhaskar_group/rho9

# 从 225 开始，处理接下来的 200 个（到 425）
nohup python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/rho9/aadhitya_v1_val \
    --knn 5 \
    --start_from 225 \
    --max_embryos 200 \
    > tphate_batch1_continue.log 2>&1 &

tail -f tphate_batch1_continue.log
```

## 注意
- DPI 降低到 150，图像质量会稍微降低，但文件大小显著减少
- 如果仍有配额问题，可以进一步降低到 100
- 使用 `--skip_existing` 参数可以跳过已存在的文件（如果脚本支持）






