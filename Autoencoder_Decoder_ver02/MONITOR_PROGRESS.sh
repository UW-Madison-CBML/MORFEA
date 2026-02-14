#!/bin/bash
# 监控 T-PHATE 生成进度

OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
LOG_FILE="/staging/groups/bhaskar_group/rho9/tphate_run.log"

echo "=== 监控 T-PHATE 生成进度 ==="
echo "按 Ctrl+C 停止"
echo ""

# 获取初始数量
LAST_COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
echo "初始数量: $LAST_COUNT / 704"
echo ""

while true; do
    # 获取当前数量
    CURRENT_COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    CURRENT_TIME=$(date '+%H:%M:%S')
    
    # 计算增加的数量
    INCREASED=$((CURRENT_COUNT - LAST_COUNT))
    
    # 显示状态
    if [ $INCREASED -gt 0 ]; then
        echo "[$CURRENT_TIME] ✓ $CURRENT_COUNT / 704 (+$INCREASED)"
    else
        echo "[$CURRENT_TIME] $CURRENT_COUNT / 704 (无变化)"
    fi
    
    # 如果数量增加了，显示最后处理的胚胎
    if [ $INCREASED -gt 0 ] && [ $CURRENT_COUNT -gt 0 ]; then
        LAST_EMBRYO=$(ls -t "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | head -1 | xargs -n1 basename | sed 's/_tphate\.png//')
        echo "    最后处理: $LAST_EMBRYO"
    fi
    
    LAST_COUNT=$CURRENT_COUNT
    
    # 如果完成了，退出
    if [ $CURRENT_COUNT -ge 704 ]; then
        echo ""
        echo "✓ 完成！所有 704 个胚胎已处理"
        break
    fi
    
    sleep 30  # 每 30 秒检查一次
done






