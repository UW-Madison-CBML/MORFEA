# 修复日志文件配额问题

## 问题
日志文件 `tphate_batch_ivf.log` 试图写入 home 目录，但 home 目录有配额限制。

## 解决方案
将日志文件也保存到 staging 目录。

## 运行命令

```bash
cd /staging/groups/bhaskar_group/rho9

# 检查当前进度
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
echo "已完成: $COUNT 个 plots"

# 使用 staging 目录的日志文件
LOG_FILE="/staging/groups/bhaskar_group/rho9/tphate_batch_ivf.log"

nohup python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/ivf/aadhitya_v1_val \
    --knn 5 \
    --start_from $COUNT \
    --max_embryos 200 \
    > "$LOG_FILE" 2>&1 &

# 查看日志
tail -f "$LOG_FILE"
```

**关键**：使用绝对路径 `"$LOG_FILE"` 指向 staging 目录，而不是 home 目录。






