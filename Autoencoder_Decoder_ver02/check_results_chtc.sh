#!/bin/bash
# 在CHTC上运行此脚本检查结果

echo "=========================================="
echo "CHTC任务结果检查"
echo "=========================================="
echo ""

# 1. 任务状态
echo "1. 当前任务状态："
condor_q | head -5
echo ""

# 2. 结果检查
echo "2. 结果检查："
RESULT_DIR="/staging/groups/bhaskar_group/rho9/model_latents/v1_baseline"
if [ -d "$RESULT_DIR" ]; then
    echo "✓ 结果目录存在"
    EMBRYO_COUNT=$(ls -1 "$RESULT_DIR/latents"/*.npy 2>/dev/null | wc -l)
    echo "  已处理embryo数量: $EMBRYO_COUNT / 704"
    
    if [ -f "$RESULT_DIR/metadata.json" ]; then
        echo "  Metadata存在"
        SUCCESS=$(grep -o '"successful":[0-9]*' "$RESULT_DIR/metadata.json" | grep -o '[0-9]*' || echo "0")
        FAILED=$(grep -o '"failed":[0-9]*' "$RESULT_DIR/metadata.json" | grep -o '[0-9]*' || echo "0")
        echo "  成功: $SUCCESS"
        echo "  失败: $FAILED"
    fi
    
    echo ""
    echo "  最新的5个文件："
    ls -lht "$RESULT_DIR/latents"/*.npy 2>/dev/null | head -5 | awk '{print "    " $9 " (" $5 ")"}'
else
    echo "✗ 结果目录不存在"
fi
echo ""

# 3. 日志检查
echo "3. 输出日志（最后20行）："
if [ -f ~/logs/extract_latents_v1_baseline.out ]; then
    tail -20 ~/logs/extract_latents_v1_baseline.out
else
    echo "  输出日志不存在"
fi
echo ""

echo "4. 错误日志（最后20行）："
if [ -f ~/logs/extract_latents_v1_baseline.err ]; then
    tail -20 ~/logs/extract_latents_v1_baseline.err
else
    echo "  错误日志不存在"
fi
echo ""

echo "5. 任务历史（最近3个）："
condor_history -limit 3 | head -20
echo ""

echo "=========================================="

