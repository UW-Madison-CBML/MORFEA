#!/bin/bash
# 检查正在运行的 T-PHATE 生成作业

echo "=== 检查进程状态 ==="
ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep
echo ""

echo "=== 查看日志（最后 20 行）==="
if [ -f /staging/groups/bhaskar_group/rho9/tphate_run.log ]; then
    tail -20 /staging/groups/bhaskar_group/rho9/tphate_run.log
else
    echo "日志文件不存在"
fi
echo ""

echo "=== 检查进度 ==="
OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
if [ -d "$OUTPUT_DIR/tphate_plots" ]; then
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    echo "已生成: $COUNT / 704 T-PHATE plots"
    echo "进度: $(( COUNT * 100 / 704 ))%"
    echo ""
    
    if [ $COUNT -gt 0 ]; then
        echo "最后处理的 5 个胚胎:"
        ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -5 | xargs -n1 basename | sed 's/_tphate\.png//'
    fi
else
    echo "输出目录不存在"
fi

echo ""
echo "=== 实时查看日志 ==="
echo "运行: tail -f /staging/groups/bhaskar_group/rho9/tphate_run.log"






