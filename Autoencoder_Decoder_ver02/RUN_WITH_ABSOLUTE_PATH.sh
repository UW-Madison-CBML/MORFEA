#!/bin/bash
# 使用绝对路径运行（保存到 staging）

cd /staging/groups/bhaskar_group/rho9

# 停止当前进程
pkill -f "generate_tphate_for_aadhitya.py" 2>/dev/null
sleep 2

echo "=== 使用绝对路径运行（保存到 staging）==="
echo ""

# 第一批：处理胚胎 183-382 (200个)
# 使用绝对路径输出到 staging
nohup python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/rho9/aadhitya_v1_val \
    --knn 5 \
    --start_from 182 \
    --max_embryos 200 \
    > tphate_batch1.log 2>&1 &

echo "✓ 作业已启动，PID: $!"
echo ""
echo "输出目录: /staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
echo "查看日志: tail -f tphate_batch1.log"






