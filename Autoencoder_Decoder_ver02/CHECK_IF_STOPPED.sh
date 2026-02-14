#!/bin/bash
# 检查进程是否停止或卡住

echo "=== 检查进程状态 ==="
ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep
if [ $? -ne 0 ]; then
    echo "⚠️  进程不在运行"
else
    echo "✓ 进程正在运行"
fi

echo ""
echo "=== 查看日志最后20行 ==="
tail -20 /tmp/tphate_run.log 2>/dev/null || echo "日志文件不存在或为空"

echo ""
echo "=== 当前进度 ==="
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
echo "已完成: $COUNT / 704 plots"

echo ""
echo "=== 最后处理的胚胎 ==="
ls -t /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | head -5 | xargs -n1 basename | sed 's/_tphate\.png//'






