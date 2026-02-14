# 处理 CPU 时间限制错误

## ❌ 错误原因

**"CPU time limit exceeded"** 表示：
- 你在登录节点上运行时间太长
- CHTC 登录节点有 CPU 时间限制（防止用户占用太多资源）
- 脚本被系统强制终止了

## ✅ 解决方案

### 方案 1: 使用 HTCondor 提交作业（推荐）

这是最好的方法，因为：
- 在专用计算节点运行（没有时间限制）
- 不占用登录节点资源
- 断开连接后继续运行

**步骤**：

```bash
# 1. 检查已生成的结果
ls -lh aadhitya_v1_val/tphate_plots/ | wc -l
ls -lh aadhitya_v1_val/curvature_plots/ | wc -l

# 2. 查看已处理了哪些胚胎
ls aadhitya_v1_val/tphate_plots/*.png | sed 's/.*\///; s/_tphate\.png//' > processed_embryos.txt

# 3. 修改脚本，跳过已处理的胚胎（或者使用 HTCondor 从断点继续）

# 4. 使用 HTCondor 提交
condor_submit generate_tphate.sub
```

### 方案 2: 使用 nohup 在后台运行（临时方案）

如果不想用 HTCondor，可以用 nohup，但仍有风险：

```bash
# 后台运行，即使断开连接也继续
nohup python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_val \
    --knn 5 > tphate_run.log 2>&1 &

# 查看进程
jobs

# 查看输出
tail -f tphate_run.log
```

⚠️ **注意**：即使使用 nohup，登录节点的 CPU 时间限制仍然存在，可能再次被终止。

### 方案 3: 分批处理

处理一部分胚胎后，检查结果，再继续：

```bash
# 先处理前 200 个
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_val_batch1 \
    --max_embryos 200

# 然后处理接下来 200 个（需要修改脚本跳过已处理的）
```

## 🔍 检查当前进度

```bash
# 查看已处理了多少胚胎
ls aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l
ls aadhitya_v1_val/curvature_plots/*.png 2>/dev/null | wc -l

# 查看最后处理的胚胎
ls -t aadhitya_v1_val/tphate_plots/*.png | head -1
```

## 💡 最佳实践建议

对于处理 **704 个胚胎**，强烈建议：

1. ✅ **使用 HTCondor 提交作业**（最可靠）
2. ⚠️ **不要直接在登录节点运行长时间任务**
3. ✅ **分批处理**（如果需要）

## 🚀 快速修复：使用 HTCondor

```bash
# 1. 确保 submit 文件存在
ls -lh generate_tphate.sub

# 2. 创建日志目录
mkdir -p logs

# 3. 提交作业
condor_submit generate_tphate.sub

# 4. 监控
condor_q -submitter rho9
condor_tail -f <ClusterID>
```






