# 重命名脚本和清理

## 在 CHTC 上运行

```bash
cd /staging/groups/bhaskar_group/rho9

# 1. 重命名脚本文件
mv generate_tphate_for_aadhitya.py generate_tphate_plots.py

# 2. 删除 aadhitya_v1_test 目录
rm -rf aadhitya_v1_test

# 3. 确认清理
find . -iname "*aadhitya*" 2>/dev/null

# 4. 检查配额
quota -s
```

## 更新运行命令

重命名后，需要使用新的脚本名：

```bash
# 使用新的脚本名
python3 generate_tphate_plots.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/rho9/v1_baseline_tphate \
    --knn 5 \
    --start_from 227 \
    > /tmp/tphate_run.log 2>&1 &
```






