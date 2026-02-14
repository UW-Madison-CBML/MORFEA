# 🚀 从Home目录提交任务（使用Staging文件）

## ⚠️ 重要限制

CHTC不允许从`/staging/`目录提交任务，必须从`/home`目录提交。

## ✅ 解决方案

使用 `extract_latents_from_home.sub`，它：
- 从home目录提交
- 使用staging目录的文件（绝对路径）
- 输出保存到staging目录

## 📋 步骤

### 1. 在CHTC上，进入home目录

```bash
cd ~
# 或
cd /home/rho9
```

### 2. 复制submit文件到home（如果需要）

```bash
cp /staging/groups/bhaskar_group/rho9/extract_latents_from_home.sub ~/
```

或者直接创建：

```bash
cd ~
nano extract_latents_from_home.sub
```

### 3. 编辑submit文件，添加任务

```bash
nano extract_latents_from_home.sub
```

在文件末尾添加：

```bash
checkpoint = /staging/groups/bhaskar_group/rho9/checkpoints/checkpoint_epoch_50.pt
model_version = v1_baseline
queue
```

### 4. 确保logs目录存在（在staging）

```bash
mkdir -p /staging/groups/bhaskar_group/rho9/logs
```

### 5. 从home目录提交

```bash
cd ~
condor_submit extract_latents_from_home.sub
```

### 6. 检查状态

```bash
condor_q

# 查看输出（在staging）
tail -f /staging/groups/bhaskar_group/rho9/logs/extract_latents_v1_baseline.out
```

## 📂 输出位置

结果会保存在staging目录：
```
/staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/
  ├── checkpoint.pt
  ├── latents/
  │   ├── embryo_*.npy
  └── metadata.json
```

## 🔍 关键点

1. **提交位置**：必须在`/home/rho9`目录
2. **文件路径**：使用绝对路径指向staging：`/staging/groups/bhaskar_group/rho9/...`
3. **输出路径**：使用`transfer_output_remaps`保存到staging
4. **日志路径**：也保存到staging的logs目录

## 💡 快速命令

```bash
# 1. 进入home
cd ~

# 2. 创建submit文件（如果还没有）
cat > extract_latents_from_home.sub << 'EOF'
# [粘贴extract_latents_from_home.sub的内容]
EOF

# 3. 编辑并添加任务
nano extract_latents_from_home.sub

# 4. 确保logs目录存在
mkdir -p /staging/groups/bhaskar_group/rho9/logs

# 5. 提交
condor_submit extract_latents_from_home.sub
```

