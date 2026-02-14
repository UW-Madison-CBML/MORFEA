# 🚀 在CHTC上运行 Extract Latent Trajectories

## 📋 步骤

### 1. 准备文件

确保以下文件在CHTC上：
- `extract_all_latent_trajectories.py`
- `extract_latents.sh`
- `extract_latents.sub`
- `model.py`
- `dataset_ivf.py`
- `build_index.py`
- `index.csv`
- `checkpoints/checkpoint_epoch_50.pt` (或你的checkpoint)

### 2. 编辑 submit 文件

编辑 `extract_latents.sub`，设置你的checkpoint和model version：

```bash
# 在文件末尾添加：
checkpoint = checkpoints/checkpoint_epoch_50.pt
model_version = v1_baseline
queue
```

如果要运行多个模型版本：

```bash
checkpoint = checkpoints/checkpoint_epoch_50.pt
model_version = v1_baseline
queue

checkpoint = checkpoints/checkpoint_epoch_50.pt
model_version = v2_no_smooth
queue
```

### 3. 创建logs目录

```bash
mkdir -p logs
```

### 4. 提交任务

```bash
condor_submit extract_latents.sub
```

### 5. 检查状态

```bash
# 查看队列
condor_q

# 查看详细信息
condor_q -better-analyze <job_id>

# 查看输出
tail -f logs/extract_latents_v1_baseline.out
tail -f logs/extract_latents_v1_baseline.err
```

### 6. 下载结果

任务完成后，结果会在 `model_latents/` 目录中：

```bash
# 查看结果
ls -lh model_latents/v1_baseline/

# 下载到本地（从CHTC）
# 结果会自动transfer回来，或者你可以手动下载
```

## 📂 输出结构

```
model_latents/
  v1_baseline/
    checkpoint.pt              # 模型checkpoint (复制)
    latents/
      embryo_ZS435-5.npy      # Latent trajectory [T, latent_dim]
      embryo_RS363-7.npy
      ...
    metadata.json              # 提取的元数据
```

## ⚙️ 配置选项

### 使用GPU（可选）

在 `extract_latents.sub` 中取消注释：

```bash
+WantGPU = true
request_gpus = 1
```

### 调整资源

如果需要更多资源，修改 `extract_latents.sub`：

```bash
request_memory = 16GB  # 增加内存
request_disk = 50GB    # 增加磁盘空间
```

### 限制embryo数量（测试用）

修改 `extract_latents.sh`，在python命令后添加：

```bash
--max_embryos 10
```

## 🔍 故障排除

### Checkpoint找不到

确保checkpoint在 `transfer_input_files` 中：
```bash
transfer_input_files = ..., $(checkpoint)
```

### index.csv找不到

确保 `index.csv` 在 `transfer_input_files` 中。

### Python包缺失

在CHTC上，你可能需要：
1. 使用conda环境
2. 或安装包：`pip3 install --user torch numpy pandas pillow`

### 内存不足

增加 `request_memory` 在 `.sub` 文件中。

## 📝 完整示例

```bash
# 1. 上传文件到CHTC
scp extract_all_latent_trajectories.py extract_latents.* model.py dataset_ivf.py build_index.py index.csv rho9@ap2001.chtc.wisc.edu:~/ivf_repo/

# 2. 上传checkpoint
scp checkpoints/checkpoint_epoch_50.pt rho9@ap2001.chtc.wisc.edu:~/ivf_repo/checkpoints/

# 3. SSH到CHTC
ssh rho9@ap2001.chtc.wisc.edu

# 4. 进入目录
cd ~/ivf_repo

# 5. 创建logs目录
mkdir -p logs

# 6. 编辑submit文件
nano extract_latents.sub
# 添加：
# checkpoint = checkpoints/checkpoint_epoch_50.pt
# model_version = v1_baseline
# queue

# 7. 提交
condor_submit extract_latents.sub

# 8. 监控
condor_q
watch -n 5 condor_q

# 9. 完成后下载结果
# 结果会自动transfer回来，或手动下载
```

## 💡 提示

1. **测试先**：先用 `--max_embryos 2` 测试
2. **检查日志**：经常查看 `.out` 和 `.err` 文件
3. **资源使用**：如果任务失败，检查是否需要更多内存/磁盘
4. **GPU**：如果使用GPU，确保 `.sub` 文件中启用了GPU请求

