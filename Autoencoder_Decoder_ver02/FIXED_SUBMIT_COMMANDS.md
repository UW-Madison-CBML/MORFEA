# 修复后的提交命令

## ✅ 正确的做法

HTCondor 不允许 `initialdir` 指向 staging，所以需要：

1. **从 home 目录提交**
2. **使用绝对路径指向脚本**
3. **日志保存在 home 目录**

### 在 CHTC 上运行：

```bash
# 1. 切换到 home 目录
cd ~

# 2. 创建日志目录
mkdir -p logs

# 3. 创建 submit 文件（不使用 initialdir）
cat > generate_tphate.sub << 'EOF'
universe = vanilla
log = $(HOME)/logs/generate_tphate_$(Cluster).log
output = $(HOME)/logs/generate_tphate_$(Cluster).out
error = $(HOME)/logs/generate_tphate_$(Cluster).err
getenv = True
executable = /usr/bin/python3
arguments = /staging/groups/bhaskar_group/rho9/generate_tphate_for_aadhitya.py --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv --output_base /staging/groups/bhaskar_group/rho9/aadhitya_v1_val --knn 5 --skip_existing
request_cpus = 4
request_memory = 8GB
request_disk = 10GB
+MaxJobRuntime = 86400
should_transfer_files = NO
queue 1
EOF

# 4. 提交作业
condor_submit generate_tphate.sub
```

### 关键点：

- ✅ **从 `~` 目录提交**
- ✅ **脚本路径**: `/staging/groups/bhaskar_group/rho9/generate_tphate_for_aadhitya.py`（绝对路径）
- ✅ **输出路径**: `/staging/groups/bhaskar_group/rho9/aadhitya_v1_val`（绝对路径）
- ✅ **日志路径**: `$(HOME)/logs/`（保存在 home 目录）
- ❌ **不使用 `initialdir`**

## 🔍 验证

```bash
# 检查 submit 文件
cat generate_tphate.sub

# 提交
condor_submit generate_tphate.sub

# 查看作业
condor_q -submitter rho9
```






