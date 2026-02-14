#!/bin/bash
# 清理空间并恢复处理

echo "============================================================"
echo "清理空间并恢复 T-PHATE 生成"
echo "============================================================"
echo ""

# 在 CHTC 上执行这些命令

cat << 'EOF'
# 1. 检查当前进度和磁盘使用
echo "=== 当前状态 ==="
OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/v1_baseline_tphate"
COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l || echo "0")
echo "已生成: $COUNT / 704 plots"
echo ""
df -h /staging/groups/bhaskar_group/rho9 | grep -E "(Filesystem|/dev/md9)"
echo ""

# 2. 停止当前进程
echo "=== 停止当前进程 ==="
pkill -f "generate_tphate_plots.py"
sleep 2

# 3. 检查是否可以删除旧的 aadhitya_v1_val 目录（如果存在）
echo "=== 检查旧目录 ==="
if [ -d "/staging/groups/bhaskar_group/rho9/aadhitya_v1_val" ]; then
    OLD_SIZE=$(du -sh /staging/groups/bhaskar_group/rho9/aadhitya_v1_val 2>/dev/null | cut -f1)
    echo "发现旧目录: aadhitya_v1_val (大小: $OLD_SIZE)"
    echo "这个目录包含之前的测试结果，可以删除以释放空间"
    echo ""
    echo "如果要删除，运行："
    echo "  rm -rf /staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
    echo "  echo '已删除旧目录'"
    echo ""
fi

# 4. 检查其他大文件/目录
echo "=== 查找其他大文件 ==="
echo "查找 >100MB 的文件/目录："
du -sh /staging/groups/bhaskar_group/rho9/* 2>/dev/null | sort -h | tail -10
echo ""

# 5. 如果需要，进一步降低DPI（从100降到75）
echo "=== 选项：进一步降低DPI ==="
echo "如果空间还是不够，可以降低DPI从100到75或50"
echo "需要修改 generate_tphate_plots.py 中的 dpi=100 为 dpi=75"
echo ""

# 6. 恢复处理（使用 --skip_existing）
echo "=== 恢复处理 ==="
CURRENT_COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l || echo "0")
echo "当前进度: $CURRENT_COUNT / 704"
echo ""
echo "继续处理（跳过已存在的）："
echo "nohup python3 /staging/groups/bhaskar_group/rho9/generate_tphate_plots.py \\"
echo "    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \\"
echo "    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \\"
echo "    --output_base /staging/groups/bhaskar_group/rho9/v1_baseline_tphate \\"
echo "    --knn 5 \\"
echo "    --skip_existing \\"
echo "    > /tmp/tphate_run.log 2>&1 &"
echo ""
echo "查看日志：tail -f /tmp/tphate_run.log"
EOF

echo ""
echo "============================================================"





