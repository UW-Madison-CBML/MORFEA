# 在 CHTC 上运行 T-PHATE 生成脚本

## 📋 步骤

### 方法 1: 上传脚本后运行（推荐）

```bash
# 1. 在本地，上传脚本
cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"
bash UPLOAD_TPHATE_SCRIPT.sh

# 2. SSH 到 CHTC
ssh rho9@ap2001.chtc.wisc.edu

# 3. 运行脚本
cd /staging/groups/bhaskar_group/rho9
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_val \
    --val_set_file validation_set.txt
```

### 方法 2: 直接在 CHTC 上创建脚本

```bash
# SSH 到 CHTC
ssh rho9@ap2001.chtc.wisc.edu

# 创建脚本文件（使用 cat 或 nano）
cd /staging/groups/bhaskar_group/rho9
# 然后复制脚本内容（使用你本地编辑器打开 generate_tphate_for_aadhitya.py，复制内容）
```

## 🔍 验证脚本是否存在

在 CHTC 上运行：

```bash
ls -lh /staging/groups/bhaskar_group/rho9/generate_tphate_for_aadhitya.py
```

## ⚠️ 注意事项

1. **脚本必须在 CHTC 上运行**，因为数据文件在 `/staging/groups/bhaskar_group/ivf/`
2. 需要先安装 `tphate`：
   ```bash
   pip install --user tphate
   ```
3. 如果你有 validation set 列表，需要先创建文件：
   ```bash
   # 创建 validation_set.txt（每行一个 cell_id）
   echo "cell_id_1" > validation_set.txt
   echo "cell_id_2" >> validation_set.txt
   # ...
   ```

## 📝 快速测试（不指定 validation set）

如果你想先测试所有胚胎：

```bash
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_test \
    --max_embryos 5  # 先测试 5 个胚胎
```






