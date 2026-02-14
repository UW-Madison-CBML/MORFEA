# 修复 GPU 作业 Idle 问题

## 🔍 诊断

先运行这个命令查看为什么 GPU 作业也是 idle：

```bash
condor_q -submitter rho9 -better-analyze
```

可能的原因：
1. GPU 节点不可用
2. 资源要求太高
3. 仍然有 FileSystemDomain 限制

## ✅ 解决方案 1: 检查 GPU 节点

```bash
# 检查 GPU 节点可用性
condor_status -constraint 'GPUs > 0' -format "Name: %s\n" Name -format "GPUs: %d\n" GPUs -format "State: %s\n" State | head -20

# 检查 bhaskar GPU 节点
condor_status -constraint 'Machine == "bhaskargpu4000.chtc.wisc.edu"'
```

## ✅ 解决方案 2: 使用普通 CPU 节点（推荐）

如果 GPU 节点不可用，使用这个版本（移除 GPU 要求，降低资源）：

```bash
cd ~

cat > generate_tphate_no_gpu.sub << 'EOF'
universe = vanilla
log = $(HOME)/logs/generate_tphate_$(Cluster).log
output = $(HOME)/logs/generate_tphate_$(Cluster).out
error = $(HOME)/logs/generate_tphate_$(Cluster).err
getenv = True

executable = /usr/bin/python3
arguments = /staging/groups/bhaskar_group/rho9/generate_tphate_for_aadhitya.py --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv --output_base /staging/groups/bhaskar_group/rho9/aadhitya_v1_val --knn 5 --skip_existing

request_cpus = 2
request_memory = 4GB
request_disk = 5GB
+MaxJobRuntime = 86400
should_transfer_files = NO

# 允许在任何有 staging 访问的节点运行
Requirements = TARGET.HasCHTCStaging == true

queue 1
EOF

# 移除旧的 GPU 作业
condor_rm 2855939

# 提交新作业
condor_submit generate_tphate_no_gpu.sub
```

## ✅ 解决方案 3: 继续在登录节点运行

如果 HTCondor 一直有问题，可以继续在登录节点运行：

```bash
cd /staging/groups/bhaskar_group/rho9

# 使用 nohup 在后台运行
nohup python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_val \
    --knn 5 \
    --skip_existing \
    > tphate_run.log 2>&1 &

# 查看输出
tail -f tphate_run.log
```

## 📊 先诊断

运行 `condor_q -submitter rho9 -better-analyze` 查看具体原因，然后决定用哪个方案。






