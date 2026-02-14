# 重新开始运行

## 清理完成

现在使用更合适的命名重新开始。

## 运行命令

```bash
cd /staging/groups/bhaskar_group/rho9

# 使用更合适的输出目录名称
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/ivf/tphate_results \
    --knn 5 \
    > /dev/null 2>&1 &
```

## 监控进度

```bash
# 监控 plots 数量
OUTPUT_DIR="/staging/groups/bhaskar_group/ivf/tphate_results"
while true; do
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l || echo "0")
    TIME=$(date '+%H:%M:%S')
    echo "[$TIME] $COUNT / 704 plots"
    
    if ! ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep > /dev/null; then
        echo "进程已停止"
        break
    fi
    
    sleep 30
done
```

## 输出位置

- T-PHATE plots: `/staging/groups/bhaskar_group/ivf/tphate_results/tphate_plots/`
- Curvature plots: `/staging/groups/bhaskar_group/ivf/tphate_results/curvature_plots/`






