# T-PHATE Plot 生成完成 - 下一步

## ✅ 当前状态

已成功生成 5 个测试胚胎的 plots：
- **T-PHATE plots**: `aadhitya_v1_test/tphate_plots/` (5 个文件)
- **Curvature plots**: `aadhitya_v1_test/curvature_plots/` (5 个文件)

## 🔍 检查结果

在 CHTC 上运行：

```bash
# 查看生成的文件
ls -lh aadhitya_v1_test/tphate_plots/
ls -lh aadhitya_v1_test/curvature_plots/

# 查看文件大小
du -sh aadhitya_v1_test/tphate_plots
du -sh aadhitya_v1_test/curvature_plots
```

## 📊 下一步选项

### 选项 1: 处理所有胚胎

如果你想处理所有 704 个胚胎：

```bash
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_all
```

**注意**: 这可能需要较长时间（704 个胚胎 × T-PHATE 计算时间）

### 选项 2: 只处理 Validation Set

如果你有 validation set 列表：

```bash
# 方法 1: 使用文本文件（每行一个 cell_id）
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_val \
    --val_set_file validation_set.txt

# 方法 2: 使用 CSV 文件
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_val \
    --val_set_csv validation_set.csv
```

### 选项 3: 分批处理

如果一次性处理所有胚胎时间太长，可以分批：

```bash
# 第一批：胚胎 1-100
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_batch1 \
    --max_embryos 100
```

## 📥 下载结果到本地

生成完成后，可以下载到本地查看：

```bash
# 在本地运行
cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

# 下载测试结果
scp -r rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/aadhitya_v1_test ./

# 或下载完整结果
scp -r rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/aadhitya_v1_all ./
```

## 💡 建议

1. **先检查测试结果**: 查看生成的 5 个 plot，确认格式和内容正确
2. **确定 validation set**: 如果有 validation set 定义，优先处理 validation set
3. **考虑计算时间**: 处理所有 704 个胚胎可能需要几个小时，考虑使用 HTCondor 提交作业






