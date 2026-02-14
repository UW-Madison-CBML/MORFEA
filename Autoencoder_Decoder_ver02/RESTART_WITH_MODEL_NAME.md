# 使用模型名称重新开始运行

## 输出目录命名

使用包含模型版本的名称：`v1_baseline_tphate`

## 运行命令

```bash
cd /staging/groups/bhaskar_group/rho9

# 使用模型版本名称
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/ivf/v1_baseline_tphate \
    --knn 5 \
    > /dev/null 2>&1 &

# 监控进度
OUTPUT_DIR="/staging/groups/bhaskar_group/ivf/v1_baseline_tphate"
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

- T-PHATE plots: `/staging/groups/bhaskar_group/ivf/v1_baseline_tphate/tphate_plots/`
- Curvature plots: `/staging/groups/bhaskar_group/ivf/v1_baseline_tphate/curvature_plots/`






