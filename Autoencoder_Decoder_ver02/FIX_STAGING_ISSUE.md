# 修复 HTCondor 提交错误

## ❌ 错误原因

HTCondor 不允许从 `/staging/` 或 `/projects/` 目录提交作业，必须从 `/home` 目录提交。

## ✅ 解决方案

### 方法 1: 从 home 目录提交（推荐）

```bash
# 1. 切换到 home 目录
cd ~

# 2. 创建 submit 文件（使用绝对路径）
cat > generate_tphate.sub << 'EOF'
universe = vanilla
log = logs/generate_tphate_$(Cluster).log
output = logs/generate_tphate_$(Cluster).out
error = logs/generate_tphate_$(Cluster).err
getenv = True
executable = /usr/bin/python3
arguments = generate_tphate_for_aadhitya.py --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv --output_base /staging/groups/bhaskar_group/rho9/aadhitya_v1_val --knn 5 --skip_existing
initialdir = /staging/groups/bhaskar_group/rho9
request_cpus = 4
request_memory = 8GB
request_disk = 10GB
+MaxJobRuntime = 86400
should_transfer_files = NO
queue 1
EOF

# 3. 创建日志目录
mkdir -p logs

# 4. 提交作业
condor_submit generate_tphate.sub
```

### 方法 2: 使用 initialdir（更简单）

```bash
# 在 home 目录创建 submit 文件，但指定工作目录为 staging
cd ~

cat > generate_tphate.sub << 'EOF'
universe = vanilla
log = /staging/groups/bhaskar_group/rho9/logs/generate_tphate_$(Cluster).log
output = /staging/groups/bhaskar_group/rho9/logs/generate_tphate_$(Cluster).out
error = /staging/groups/bhaskar_group/rho9/logs/generate_tphate_$(Cluster).err
getenv = True
executable = /usr/bin/python3
arguments = generate_tphate_for_aadhitya.py --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv --output_base /staging/groups/bhaskar_group/rho9/aadhitya_v1_val --knn 5 --skip_existing
initialdir = /staging/groups/bhaskar_group/rho9
request_cpus = 4
request_memory = 8GB
request_disk = 10GB
+MaxJobRuntime = 86400
should_transfer_files = NO
queue 1
EOF

mkdir -p /staging/groups/bhaskar_group/rho9/logs
condor_submit generate_tphate.sub
```






