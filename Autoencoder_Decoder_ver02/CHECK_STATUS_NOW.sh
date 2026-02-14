#!/bin/bash
# 检查当前作业状态

echo "=== 1. 检查进程是否在运行 ==="
ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep
if [ $? -ne 0 ]; then
    echo "⚠️  进程不在运行"
else
    echo "✓ 进程正在运行"
fi
echo ""

echo "=== 2. 查看日志最后 30 行 ==="
if [ -f /staging/groups/bhaskar_group/rho9/tphate_run.log ]; then
    tail -30 /staging/groups/bhaskar_group/rho9/tphate_run.log
else
    echo "⚠️  日志文件不存在"
fi
echo ""

echo "=== 3. 检查当前进度 ==="
OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
if [ -d "$OUTPUT_DIR/tphate_plots" ]; then
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    echo "已生成: $COUNT / 704 T-PHATE plots"
    echo "剩余: $(( 704 - COUNT )) 个胚胎"
    
    echo ""
    echo "最后处理的胚胎:"
    ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -3 | xargs -n1 basename | sed 's/_tphate\.png//'
else
    echo "输出目录不存在"
fi

echo ""
echo "=== 提示 ==="
echo "如果进程在运行但数量没增加，可能是："
echo "  1. 正在重新处理已完成的胚胎（因为没用 --skip_existing）"
echo "  2. 正在处理一个很大的胚胎（需要较长时间）"
echo ""
echo "查看实时日志: tail -f /staging/groups/bhaskar_group/rho9/tphate_run.log"






