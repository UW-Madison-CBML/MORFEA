# 在 CHTC 上设置 T-PHATE 生成

## 🚀 快速设置步骤

### 1. 创建 submit 文件

在 CHTC 上运行：

```bash
cd /staging/groups/bhaskar_group/rho9

# 创建 submit 文件
cat > generate_tphate.sub << 'SUBMIT_EOF'
# HTCondor submit file for generating T-PHATE plots
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
SUBMIT_EOF
```

### 2. 创建日志目录

```bash
mkdir -p logs
```

### 3. 验证文件存在

```bash
ls -lh generate_tphate.sub
ls -lh generate_tphate_for_aadhitya.py
```

### 4. 提交作业

```bash
condor_submit generate_tphate.sub
```

### 5. 监控作业

```bash
# 查看作业状态
condor_q -submitter rho9

# 查看实时输出（替换 <ClusterID> 为实际 ID）
condor_tail -f <ClusterID>
```

## ⚠️ 如果 condor 命令不工作

检查 condor 是否正确配置：

```bash
# 检查 condor 是否可用
which condor_submit

# 如果找不到，可能需要加载环境
source /etc/profile.d/condor.sh

# 或者
export PATH=/usr/bin:$PATH
```

## 🔍 检查已处理的胚胎

```bash
# 查看已处理多少
ls -1 aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l

# 查看最后处理的胚胎
ls -t aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | head -5
```






