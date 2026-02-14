#!/bin/bash
# 停止进程并使用正确的绝对路径重新运行

cd /staging/groups/bhaskar_group/rho9

echo "=== 停止所有相关进程 ==="
pkill -f "generate_tphate_for_aadhitya.py"
sleep 2

echo "=== 检查进程是否已停止 ==="
if ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep > /dev/null; then
    echo "⚠️  还有进程在运行，强制停止..."
    pkill -9 -f "generate_tphate_for_aadhitya.py"
    sleep 1
else
    echo "✓ 所有进程已停止"
fi

echo ""
echo "=== 使用绝对路径重新运行 ==="
echo "输出将保存到: /staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
echo ""

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
echo "查看日志: tail -f tphate_batch1.log"
echo "检查输出目录: ls -lh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/ | head -10"






