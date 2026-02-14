# 分批运行 T-PHATE 生成

## 当前状态
- 已完成的胚胎: **182 个**
- 总胚胎数: **704 个**
- 剩余: **522 个**

## 方法：使用 --start_from 和 --max_embryos 参数

### 第一步：更新脚本

先上传更新后的脚本（添加了 `--start_from` 参数）：

```bash
# 从本地上传
cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"
scp generate_tphate_for_aadhitya.py rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/
```

### 第二步：分批运行

```bash
cd /staging/groups/bhaskar_group/rho9

# 第一批：处理胚胎 183-382 (200个)
# start_from=182 表示跳过前182个（索引从0开始，所以182是第183个）
nohup python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_val \
    --knn 5 \
    --start_from 182 \
    --max_embryos 200 \
    > tphate_batch1.log 2>&1 &

# 查看进度
tail -f tphate_batch1.log
```

### 后续批次

等待第一批完成后，运行：

```bash
# 第二批：处理胚胎 383-582 (200个)
nohup python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_val \
    --knn 5 \
    --start_from 382 \
    --max_embryos 200 \
    > tphate_batch2.log 2>&1 &

# 第三批：处理胚胎 583-704 (122个，剩余的全部)
nohup python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_val \
    --knn 5 \
    --start_from 582 \
    > tphate_batch3.log 2>&1 &
```

## 检查进度

```bash
# 查看当前总数
ls -1 aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l

# 查看最后处理的胚胎
ls -t aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | head -5
```

## 优势

- ✅ **避免重新处理已完成的胚胎**
- ✅ **每批处理200个，避免CPU时间限制**
- ✅ **可以分别监控每批的进度**
- ✅ **如果某批失败，可以重新运行那一批**






