# 清理并继续运行

## 步骤

### 1. 清理 home 目录释放空间

```bash
# 删除 .cache（最安全，124M）
rm -rf ~/.cache

# 删除压缩文件（如果不需要，555M）
rm ~/ivf-embryo-analysis-Raffael.tgz

# 检查配额
quota -s
```

### 2. 继续运行

```bash
cd /staging/groups/bhaskar_group/rho9

# 检查当前进度
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
echo "当前进度: $COUNT / 704"

# 继续运行
python3 generate_tphate_plots.py \
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
    
    if ! ps aux | grep "generate_tphate_plots.py" | grep -v grep > /dev/null; then
        echo "进程已停止，当前进度: $NEW_COUNT"
        break
    fi
    
    sleep 30
done
```

## 预期

删除 .cache 和压缩文件后，可以释放约 679M，配额应该会降到约 39242M，有约 1.7GB 可用空间，应该可以继续处理更多胚胎。






