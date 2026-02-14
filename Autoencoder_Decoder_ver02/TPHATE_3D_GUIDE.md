# 3D T-PHATE 可视化指南

使用 epoch 50 checkpoint 的 latent vectors 进行 3D T-PHATE 可视化。

## 前置要求

1. **phate 库**：标准的 T-PHATE 实现
   ```bash
   pip install phate
   ```

2. **其他依赖**：
   ```bash
   pip install matplotlib numpy pandas scikit-learn
   ```

## 在 CHTC 上运行（推荐）

### 步骤 1: 上传脚本

```bash
# 在本地 Mac 执行
cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"
scp extract_latents_epoch50.py tphate_3d_visualization.py run_tphate_3d_chtc.sh rho9@ap2001.chtc.wisc.edu:~/ivf_repo/
```

### 步骤 2: 在 CHTC 上运行

```bash
# SSH 到 CHTC
ssh rho9@ap2001.chtc.wisc.edu
cd ~/ivf_repo

# 安装 phate 库（如果需要）
pip install --user phate

# 运行完整流程
chmod +x run_tphate_3d_chtc.sh
./run_tphate_3d_chtc.sh
```

### 或者分步运行

```bash
# 1. 提取 latent vectors
python3 extract_latents_epoch50.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --index_csv index.csv \
    --output_dir latents_epoch50 \
    --batch_size 8 \
    --use_z_seq

# 2. 应用 3D T-PHATE
python3 tphate_3d_visualization.py \
    --latents_file latents_epoch50/latents_z_seq_epoch50.npy \
    --metadata_file latents_epoch50/latents_metadata_epoch50.json \
    --output_dir tphate_3d_results \
    --n_embryos 10 \
    --knn 5 \
    --decay 40
```

## 输出文件

### Latent Vectors (`latents_epoch50/`)
- `latents_z_seq_epoch50.npy` - 完整序列的 latent vectors [N, T, hidden_dim]
- `latents_z_last_epoch50.npy` - 最后一个时间步的 latent vectors [N, hidden_dim]
- `latents_metadata_epoch50.json` - 元数据（cell_id, start_idx 等）
- `latents_info_epoch50.csv` - CSV 格式的元数据

### 3D T-PHATE 可视化 (`tphate_3d_results/`)
- `tphate_3d_comparison.png` - 所有胚胎的 3D 轨迹对比图
- `tphate_3d_embryo_{cell_id}.png` - 单个胚胎的 3D 轨迹图

## 参数说明

### extract_latents_epoch50.py
- `--checkpoint`: Checkpoint 文件路径（默认: `checkpoints/checkpoint_epoch_50.pt`）
- `--index_csv`: 数据集索引文件（默认: `index.csv`）
- `--output_dir`: 输出目录（默认: `latents_epoch50`）
- `--batch_size`: Batch size（默认: 8）
- `--use_z_seq`: 提取完整序列（用于 t-PHATE）

### tphate_3d_visualization.py
- `--latents_file`: Latent vectors 文件路径
- `--metadata_file`: 元数据文件路径
- `--output_dir`: 输出目录（默认: `tphate_3d_results`）
- `--n_embryos`: 要可视化的胚胎数量（默认: 10）
- `--knn`: k-NN 参数（默认: 5）
- `--decay`: Decay 参数（默认: 40）

## 下载结果

```bash
# 在本地 Mac 执行
cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

# 下载 latent vectors
scp -r rho9@ap2001.chtc.wisc.edu:~/ivf_repo/latents_epoch50 ./

# 下载可视化结果
scp -r rho9@ap2001.chtc.wisc.edu:~/ivf_repo/tphate_3d_results ./
```

## 可视化说明

3D T-PHATE 图显示：
- **X, Y, Z 轴**：T-PHATE 的三个主要组件
- **轨迹线**：胚胎发育的时间轨迹
- **颜色**：不同胚胎用不同颜色表示
- **起点/终点**：圆形标记起点，方形标记终点

## 注意事项

1. **模型架构**：脚本会自动检测 CHTC 上的模型结构（Encoder + Decoder）
2. **内存使用**：提取所有 latent vectors 可能需要较多内存
3. **计算时间**：T-PHATE 计算可能需要几分钟，取决于数据量

