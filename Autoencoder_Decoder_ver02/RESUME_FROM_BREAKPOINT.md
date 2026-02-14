# 从断点继续运行

## 🔍 当前情况

脚本在登录节点上运行到第 183 个胚胎时被终止：
- **错误**: "CPU time limit exceeded"
- **原因**: 登录节点有 CPU 时间限制
- **已处理**: 约 182 个胚胎

## ✅ 解决方案：从断点继续

我已经添加了 `--skip_existing` 参数，可以跳过已处理的胚胎。

### 方法 1: 使用 HTCondor（推荐）

```bash
# 1. 检查已处理的胚胎
ls -lh aadhitya_v1_val/tphate_plots/ | wc -l

# 2. 使用 HTCondor 提交（会自动跳过已处理的）
condor_submit generate_tphate.sub

# 3. 监控作业
condor_q -submitter rho9
condor_tail -f <ClusterID>
```

**优势**：
- ✅ 在计算节点运行（无时间限制）
- ✅ 自动跳过已处理的胚胎
- ✅ 可以长时间运行

### 方法 2: 在登录节点继续（不推荐，可能再次被限制）

```bash
# 继续运行，跳过已处理的胚胎
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base aadhitya_v1_val \
    --skip_existing
```

⚠️ **警告**: 即使使用 `--skip_existing`，在登录节点运行仍可能再次被限制。

## 🔍 检查进度

```bash
# 查看已处理多少胚胎
PROCESSED=$(ls -1 aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
REMAINING=$((704 - PROCESSED))
echo "已处理: $PROCESSED / 704"
echo "剩余: $REMAINING"

# 查看最后处理的胚胎
ls -t aadhitya_v1_val/tphate_plots/*.png | head -1
```

## 💡 最佳实践

对于剩余 522 个胚胎（704 - 182 = 522），强烈建议：

1. ✅ **使用 HTCondor 提交作业**
2. ✅ **使用 `--skip_existing` 参数**
3. ✅ **检查 submit 文件设置是否正确**






