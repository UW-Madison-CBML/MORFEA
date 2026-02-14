# 使用 GPU 提交 T-PHATE 生成作业

## ✅ 优势

使用 bhaskar 的 GPU 节点：
- **更容易匹配**：GPU 节点通常更容易获得
- **可能更快**：如果 T-PHATE 库支持 GPU 加速
- **避免 FileSystemDomain 问题**：GPU 节点配置不同

## 🚀 提交步骤

```bash
cd ~
mkdir -p logs

# 创建 submit 文件
cat > generate_tphate_gpu.sub << 'EOF'
# Lab GPU settings
+WantGPULab = true
+GPUJobLength = "long"
+ProjectName = "UWMadison_BME_Bhaskar"

universe = vanilla
log = $(HOME)/logs/generate_tphate_$(Cluster).log
output = $(HOME)/logs/generate_tphate_$(Cluster).out
error = $(HOME)/logs/generate_tphate_$(Cluster).err
getenv = True

executable = /usr/bin/python3
arguments = /staging/groups/bhaskar_group/rho9/generate_tphate_for_aadhitya.py --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv --output_base /staging/groups/bhaskar_group/rho9/aadhitya_v1_val --knn 5 --skip_existing

request_gpus = 1
request_cpus = 4
request_memory = 8GB
request_disk = 10GB
+MaxJobRuntime = 86400
should_transfer_files = NO

queue 1
EOF

# 先移除旧的 idle 作业（如果有）
condor_rm 2855933

# 提交新作业
condor_submit generate_tphate_gpu.sub
```

## 🔍 检查作业状态

```bash
# 查看作业
condor_q -submitter rho9

# 查看详细信息
condor_q -submitter rho9 -better-analyze

# 查看进度
ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l
```

## 📝 注意事项

- GPU 节点通常更容易匹配，因为使用的人较少
- 即使 T-PHATE 不使用 GPU，在 GPU 节点上运行也没问题
- 使用 `--skip_existing` 会跳过已完成的 182 个胚胎






