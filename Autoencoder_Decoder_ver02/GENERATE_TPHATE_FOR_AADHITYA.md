# 为 Aadhitya 的 Latents 生成 T-PHATE 和 Curvature Plot

## 📋 概述

**重要：此脚本直接读取已存储的 latent vectors，不进行 inference。**

这个脚本会为 Aadhitya 的 latent 数据生成：
1. **T-PHATE plots** - **3D** 轨迹图（按时间着色），保存在 `{output_base}/tphate_plots/`
2. **Curvature plots** - **3D** 轨迹图（按曲率着色），保存在 `{output_base}/curvature_plots/`

## 🚀 使用方法

### 在 CHTC 上运行：

```bash
# 1. SSH 到 CHTC
ssh rho9@ap2001.chtc.wisc.edu

# 2. 导航到工作目录
cd /staging/groups/bhaskar_group/rho9  # 或你的工作目录

# 3. 上传脚本（如果需要）
# 或者直接在 CHTC 上创建脚本

# 4. 确保安装了 tphate
pip install --user tphate

# 5. 运行脚本
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1 \
    --knn 5
```

## 📝 参数说明

- `--npy_file`: latents.npy 文件路径（默认: `/staging/groups/bhaskar_group/ivf/latents/latents.npy`）
- `--csv_file`: latents.csv 文件路径（默认: `/staging/groups/bhaskar_group/ivf/latents/latents.csv`）
- `--output_base`: 输出基础目录名（默认: `aadhitya_v1`）
- `--knn`: T-PHATE 的 k-nearest neighbors 参数（默认: 5）
- `--max_embryos`: 最大处理胚胎数（默认: None，处理所有）
- `--val_set_file`: Validation set 文件路径（每行一个 cell_id，可选）
- `--val_set_csv`: Validation set CSV 文件路径（需要包含 `cell_id` 列，可选）

## 📁 输出结构

```
aadhitya_v1/
├── tphate_plots/
│   ├── {cell_id}_tphate.png
│   ├── {cell_id}_tphate.png
│   └── ...
└── curvature_plots/
    ├── {cell_id}_curvature.png
    ├── {cell_id}_curvature.png
    └── ...
```

## ⚠️ 注意事项

1. **需要 tphate 库**: 运行前确保安装了 `tphate`
   ```bash
   pip install --user tphate
   ```

2. **处理时间**: 对于 704 个胚胎，可能需要较长时间。可以使用 `--max_embryos` 先测试少量胚胎。

3. **内存使用**: 确保有足够的内存来处理所有数据。

## 🔍 示例用法

### 1. 先测试 10 个胚胎

```bash
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_test \
    --max_embryos 10
```

### 2. 只处理 validation set（推荐）

```bash
# 如果有 validation set 列表文件（每行一个 cell_id）
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_val \
    --val_set_file validation_set.txt

# 或者如果有 validation set CSV 文件
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_val \
    --val_set_csv validation_set.csv
```

## 📊 输出文件说明

### T-PHATE Plot (`{cell_id}_tphate.png`)
- **3D** T-PHATE 轨迹图
- 按时间着色
- 标记起点（绿色）和终点（红色）
- 显示轨迹长度统计

### Curvature Plot (`{cell_id}_curvature.png`)
- **3D** T-PHATE 轨迹图
- 按曲率着色（颜色越亮表示曲率越高）
- 标记起点和终点
- 显示最大和平均曲率统计

## 🐛 故障排除

### 如果 tphate 安装失败：

```bash
# 尝试先安装依赖
pip install --user --upgrade numpy setuptools wheel
pip install --user --no-cache-dir s_gd2
pip install --user --no-cache-dir tphate
```

### 如果内存不足：

- 使用 `--max_embryos` 参数分批处理
- 或者使用 HTCondor 提交作业（需要创建 submit 文件）

## ✅ 验证输出

运行完成后，检查输出：

```bash
# 检查生成的 plot 数量
ls -lh aadhitya_v1/tphate_plots/*.png | wc -l
ls -lh aadhitya_v1/curvature_plots/*.png | wc -l

# 应该与处理的胚胎数量相同
```

