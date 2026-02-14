# 不使用日志文件监控进度

## 问题
无法创建日志文件（权限问题或配额限制）

## 解决方案
不使用日志文件，直接监控输出目录的进度

## 检查进程和进度

```bash
# 1. 检查进程是否在运行
ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep

# 2. 检查输出目录
ls -1 /staging/groups/bhaskar_group/ivf/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l

# 3. 监控进度（每30秒检查一次）
OUTPUT_DIR="/staging/groups/bhaskar_group/ivf/aadhitya_v1_val"
while true; do
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    TIME=$(date '+%H:%M:%S')
    echo "[$TIME] $COUNT plots"
    
    # 检查进程
    if ! ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep > /dev/null; then
        echo "进程已停止"
        break
    fi
    
    sleep 30
done
```

## 如果进程不在运行

如果进程已停止，重新运行（不使用日志文件）：

```bash
cd /staging/groups/bhaskar_group/rho9

COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)

# 不使用日志文件，直接运行
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/ivf/aadhitya_v1_val \
    --knn 5 \
    --start_from $COUNT \
    --max_embryos 200 \
    > /dev/null 2>&1 &

# 然后监控输出目录
```






