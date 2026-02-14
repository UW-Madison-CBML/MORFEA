#!/bin/bash
# 停止进程并总结当前状态

echo "=== 停止所有相关进程 ==="
pkill -f "generate_tphate_plots.py"
sleep 2
ps aux | grep "generate_tphate_plots.py" | grep -v grep || echo "✓ 所有进程已停止"

echo ""
echo "=== 当前状态总结 ==="
COUNT=$(ls -1 /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/*.png 2>/dev/null | wc -l)
echo "已完成: $COUNT / 704 个胚胎 ($((COUNT * 100 / 704))%)"
echo ""
echo "输出位置:"
echo "  T-PHATE plots: /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/tphate_plots/"
echo "  Curvature plots: /staging/groups/bhaskar_group/rho9/v1_baseline_tphate/curvature_plots/"
echo ""
echo "目录大小:"
du -sh /staging/groups/bhaskar_group/rho9/v1_baseline_tphate

echo ""
echo "=== 配额状态 ==="
quota -s

echo ""
echo "=== 问题 ==="
echo "配额已满（39921M / 40960M），无法继续处理剩余 $((704 - COUNT)) 个胚胎"

echo ""
echo "=== 解决方案 ==="
echo "1. 联系 CHTC 管理员增加配额（推荐，如果需要所有 704 个胚胎）"
echo "2. 只处理 validation set（如果只需要这些胚胎）"
echo "3. 接受当前结果（233 个胚胎的 plots）"






