# 生成重建示例 (Generate Reconstruction Examples)

## 概述

`generate_reconstructions.py` 脚本可以从训练好的模型中生成重建示例，展示原始图像和模型重建图像的对比。

## 使用方法

### 在本地 Mac 上运行（需要完整数据集）

```bash
cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

# 生成 10 个随机样本的重建示例
python3 generate_reconstructions.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --index_csv index.csv \
    --output_dir reconstructions \
    --num_samples 10 \
    --n_frames 10
```

### 在 CHTC 上运行（推荐）

由于数据集在 CHTC 上，建议在 CHTC 上运行：

```bash
# SSH 到 CHTC
ssh rho9@ap2001.chtc.wisc.edu

# 进入工作目录
cd ~/ivf_repo

# 上传脚本（如果还没有）
# 从本地 Mac 执行：
# scp generate_reconstructions.py rho9@ap2001.chtc.wisc.edu:~/ivf_repo/

# 运行脚本
python3 generate_reconstructions.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --index_csv index.csv \
    --output_dir reconstructions \
    --num_samples 10 \
    --n_frames 10
```

### 生成特定序列的重建

```bash
python3 generate_reconstructions.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --index_csv index.csv \
    --output_dir reconstructions \
    --cell_id AA83-7 \
    --start_idx 0 \
    --n_frames 20
```

## 参数说明

- `--checkpoint`: 模型 checkpoint 路径（默认: `checkpoints/checkpoint_epoch_50.pt`）
- `--index_csv`: 数据集索引文件（默认: `index.csv`）
- `--output_dir`: 输出目录（默认: `reconstructions/`）
- `--num_samples`: 要生成的样本数量（默认: 10）
- `--n_frames`: 每个样本显示的帧数（默认: 10）
- `--cell_id`: 特定 cell ID（可选）
- `--start_idx`: 特定 start index（可选）

## 输出

脚本会在 `reconstructions/` 目录生成 PNG 图像文件，每个文件包含：
- 上排：原始图像序列
- 下排：模型重建的图像序列
- 标题：cell_id 和 start_idx

文件名格式：`reconstruction_{cell_id}_start{start_idx}.png`

## 注意事项

1. **模型结构兼容性**: 如果遇到模型加载错误，脚本会尝试使用 `strict=False` 加载权重
2. **数据路径**: 确保 `index.csv` 中的图像路径可访问
3. **内存使用**: 生成大量样本可能需要较多内存

## 示例输出

```
Using device: cuda
Loading checkpoint from: checkpoints/checkpoint_epoch_50.pt
Model loaded successfully!
Loading dataset from: index.csv
Selected 10 random samples from 13298 total sequences

Generating reconstructions...
Reconstructing: 100%|██████████| 10/10 [00:30<00:00,  3.05s/it]
  [1/10] Saved: reconstruction_AA83-7_start0.png
  [2/10] Saved: reconstruction_BB12-3_start16.png
  ...

✅ Generated 10 reconstruction examples!
📁 Files saved in: reconstructions/
```

## 下载结果到本地

在 CHTC 上生成后，可以下载到本地：

```bash
# 在本地 Mac 执行
scp -r rho9@ap2001.chtc.wisc.edu:~/ivf_repo/reconstructions ./
```

