# 在 CHTC 上快速设置

## 🚀 一步到位：创建 submit 文件

在 CHTC 上运行以下命令（复制粘贴整个块）：

```bash
cd /staging/groups/bhaskar_group/rho9

# 创建 submit 文件（单行，避免语法错误）
cat > generate_tphate.sub << 'EOF'
universe = vanilla
log = logs/generate_tphate_$(Cluster).log
output = logs/generate_tphate_$(Cluster).out
error = logs/generate_tphate_$(Cluster).err
getenv = True
executable = /usr/bin/python3
arguments = generate_tphate_for_aadhitya.py --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv --output_base aadhitya_v1_val --knn 5 --skip_existing
request_cpus = 4
request_memory = 8GB
request_disk = 10GB
+MaxJobRuntime = 86400
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_output_files = aadhitya_v1_val
queue 1
EOF

# 创建日志目录
mkdir -p logs

# 验证文件
ls -lh generate_tphate.sub
ls -lh generate_tphate_for_aadhitya.py

# 提交作业
condor_submit generate_tphate.sub
```

## 🔍 如果 condor_submit 不工作

```bash
# 检查 condor 是否在 PATH
which condor_submit

# 如果找不到，尝试加载环境
source /etc/profile.d/condor.sh 2>/dev/null || true

# 或者使用完整路径
/usr/bin/condor_submit generate_tphate.sub
```

## 📊 检查进度

```bash
# 查看作业状态
condor_q -submitter rho9

# 查看已处理的胚胎
ls -1 aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l
```






