# 保存到 ivf 目录

## 尝试在 ivf 目录创建输出目录

```bash
cd /staging/groups/bhaskar_group/rho9

# 1. 检查 ivf 目录权限
ls -ld /staging/groups/bhaskar_group/ivf/

# 2. 尝试创建输出目录
mkdir -p /staging/groups/bhaskar_group/ivf/v1_baseline_tphate/tphate_plots
mkdir -p /staging/groups/bhaskar_group/ivf/v1_baseline_tphate/curvature_plots

# 3. 如果成功，继续运行
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
echo "当前进度: $COUNT / 704"

python3 generate_tphate_plots.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/ivf/v1_baseline_tphate \
    --knn 5 \
    --start_from $COUNT \
    > /tmp/tphate_run.log 2>&1 &
```

如果 ivf 目录没有配额限制，保存到那里应该可以避免配额问题。






