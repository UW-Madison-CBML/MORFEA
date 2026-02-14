#!/bin/bash
# 检查脚本是否卡住

cd /staging/groups/bhaskar_group/rho9

echo "=== 检查进程状态 ==="
ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep
if [ $? -ne 0 ]; then
    echo "⚠️  进程不在运行！"
else
    echo "✓ 进程正在运行"
fi
echo ""

echo "=== 查看日志最后 50 行 ==="
if [ -f tphate_batch1.log ]; then
    tail -50 tphate_batch1.log
else
    echo "⚠️  日志文件不存在"
fi
echo ""

echo "=== 检查当前进度 ==="
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
echo "当前: $COUNT / 704 plots"
echo ""

echo "=== 检查最后处理的胚胎 ==="
ls -t /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | head -5 | xargs -n1 basename | sed 's/_tphate\.png//'
echo ""

echo "=== 如果卡住了，可能原因："
echo "  1. 正在处理一个很大的胚胎（需要较长时间）"
echo "  2. 遇到错误（查看上面的日志）"
echo "  3. CPU 时间限制（但进程应该还在）"






