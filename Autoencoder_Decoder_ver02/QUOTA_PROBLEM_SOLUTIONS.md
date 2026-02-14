# 配额问题解决方案

## 当前状况
- 配额使用：39921M / 40960M (97.5%)
- 已生成：227 个胚胎（454 个 plots）
- 剩余：477 个胚胎（954 个 plots）

## 解决方案

### 方案1: 只处理 Validation Set（推荐）

如果只需要处理 validation set 的胚胎，这样可以大幅减少需要处理的胚胎数量。

```bash
# 假设有 validation_set.txt 文件
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/rho9/v1_baseline_tphate \
    --val_set_file validation_set.txt \
    --knn 5 \
    > /tmp/tphate_run.log 2>&1 &
```

### 方案2: 进一步降低 DPI 到 75 或 50

```python
# 修改脚本中的 dpi=100 改为 dpi=75 或 dpi=50
plt.savefig(save_path, dpi=75, bbox_inches='tight')
```

### 方案3: 使用 JPEG 格式（文件更小）

修改脚本使用 JPEG 格式而不是 PNG，文件大小可以减小 5-10 倍。

### 方案4: 联系管理员增加配额

如果确实需要处理所有 704 个胚胎，可能需要联系 CHTC 管理员增加配额。

### 方案5: 分批处理并压缩

处理完一批后，压缩已完成的 plots：

```bash
# 压缩已完成的 plots
cd /staging/groups/bhaskar_group/rho9/v1_baseline_tphate
tar -czf tphate_plots_227.tar.gz tphate_plots/
tar -czf curvature_plots_227.tar.gz curvature_plots/

# 删除原始文件（释放空间）
rm -rf tphate_plots/*.png curvature_plots/*.png
```

## 建议

如果只需要 validation set，使用方案1最简单。
如果需要所有胚胎，可能需要方案4（联系管理员）或方案5（分批压缩）。






