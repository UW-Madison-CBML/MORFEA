#!/bin/bash
# 检查第一批的运行状态

cd /staging/groups/bhaskar_group/rho9

echo "=== 检查进程状态 ==="
if ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep > /dev/null; then
    echo "✓ 进程正在运行"
    ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep
else
    echo "⚠️  进程不在运行"
fi
echo ""

echo "=== 查看日志（最后 30 行）==="
if [ -f tphate_batch1.log ]; then
    tail -30 tphate_batch1.log
else
    echo "⚠️  日志文件不存在（可能还在初始化）"
fi
echo ""

echo "=== 检查进度 ==="
COUNT=$(ls -1 aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
echo "当前总数: $COUNT / 704"
EXPECTED_AFTER_BATCH1=382
REMAINING=$((EXPECTED_AFTER_BATCH1 - COUNT))

if [ $COUNT -gt 182 ]; then
    echo "✓ 数量已增加！第一批正在处理"
    echo "  已完成: $((COUNT - 182)) / 200 (第一批)"
    echo "  剩余: $REMAINING 个"
    echo ""
    echo "最后处理的胚胎:"
    ls -t aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | head -3 | xargs -n1 basename | sed 's/_tphate\.png//'
elif [ $COUNT -eq 182 ]; then
    echo "数量还是 182，可能还在初始化（加载数据）..."
fi

echo ""
echo "=== 实时监控 ==="
echo "查看日志: tail -f tphate_batch1.log"
echo "检查进度: watch -n 30 'ls -1 aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l'"






