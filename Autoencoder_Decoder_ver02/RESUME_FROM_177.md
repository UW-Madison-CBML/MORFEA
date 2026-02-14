# 从断点继续运行

## 当前状态
- 已完成：177 / 704 plots
- 剩余：527 个胚胎

## 继续运行

```bash
cd /staging/groups/bhaskar_group/rho9

# 检查当前进度
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
echo "已完成: $COUNT 个 plots"

# 从断点继续（使用 --start_from）
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/rho9/v1_baseline_tphate \
    --knn 5 \
    --start_from $COUNT \
    > /tmp/tphate_run.log 2>&1 &

# 监控进度
OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/v1_baseline_tphate"
while true; do
    NEW_COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l || echo "0")
    TIME=$(date '+%H:%M:%S')
    echo "[$TIME] $NEW_COUNT / 704 plots"
    
    if ! ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep > /dev/null; then
        echo "进程已停止"
        break
    fi
    
    sleep 30
done
```

## 注意
- 使用 `--start_from $COUNT` 会跳过已完成的 177 个胚胎
- 如果再次遇到 CPU 时间限制，可以继续用同样的方法从新的断点继续






