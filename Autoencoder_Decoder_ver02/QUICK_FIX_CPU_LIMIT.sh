#!/bin/bash
# 快速修复 CPU 时间限制问题

echo "=== 处理 CPU 时间限制错误 ==="
echo ""

# 检查已处理的胚胎数量
if [ -d "aadhitya_v1_val/tphate_plots" ]; then
    PROCESSED=$(ls -1 aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l)
    echo "✓ 已处理: $PROCESSED 个胚胎"
    
    # 显示最后几个处理的胚胎
    echo ""
    echo "最后处理的 5 个胚胎:"
    ls -t aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | head -5 | sed 's/.*\///; s/_tphate\.png//'
else
    echo "⚠️  输出目录不存在"
fi

echo ""
echo "=== 解决方案 ==="
echo ""
echo "1. 使用 HTCondor 提交作业（推荐）:"
echo "   condor_submit generate_tphate.sub"
echo ""
echo "2. 或者在登录节点上继续运行（添加 --skip_existing 跳过已处理的）:"
echo "   python3 generate_tphate_for_aadhitya.py \\"
echo "       --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \\"
echo "       --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \\"
echo "       --output_base aadhitya_v1_val \\"
echo "       --skip_existing"
echo ""
echo "⚠️  注意：在登录节点运行仍可能被限制，建议使用 HTCondor"






