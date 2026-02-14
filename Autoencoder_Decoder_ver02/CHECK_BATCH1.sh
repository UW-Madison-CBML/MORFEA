#!/bin/bash
# 检查第一批的运行状态

cd /staging/groups/bhaskar_group/rho9

echo "=== 检查进程状态 ==="
ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep
echo ""

echo "=== 查看日志文件 ==="
if [ -f tphate_batch1.log ]; then
    echo "日志文件存在，大小: $(ls -lh tphate_batch1.log | awk '{print $5}')"
    echo ""
    echo "最后 30 行:"
    tail -30 tphate_batch1.log
else
    echo "⚠️  日志文件不存在"
fi

echo ""
echo "=== 检查进度 ==="
COUNT=$(ls -1 aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
echo "当前总数: $COUNT / 704"
echo "预期: 182 + 200 = 382 (第一批完成后)"
echo ""

if [ $COUNT -gt 182 ]; then
    echo "✓ 数量已增加，第一批正在处理！"
    echo ""
    echo "最后处理的胚胎:"
    ls -t aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | head -5 | xargs -n1 basename | sed 's/_tphate\.png//'
else
    echo "数量还是 182，可能还在初始化..."
fi






