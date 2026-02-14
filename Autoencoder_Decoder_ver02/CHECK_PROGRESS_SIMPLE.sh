#!/bin/bash
# 简单检查进度

OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
PROGRESS=$((COUNT - 182))
TIME=$(date '+%H:%M:%S')

echo "[$TIME] 当前进度"
echo "总 plots: $COUNT / 704"
echo "第一批进度: $PROGRESS / 200"
echo ""

if [ $COUNT -ge 382 ]; then
    echo "✓ 第一批完成！可以开始第二批了"
elif [ $COUNT -gt 182 ]; then
    REMAINING=$((382 - COUNT))
    echo "还需处理: $REMAINING 个胚胎（第一批）"
else
    echo "还在初始化..."
fi
