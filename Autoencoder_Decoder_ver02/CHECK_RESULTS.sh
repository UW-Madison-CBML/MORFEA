#!/bin/bash
# 检查CHTC任务结果

echo "=== 检查CHTC任务结果 ==="
echo ""

echo "1. 当前任务状态："
echo "---"
echo "在CHTC上运行: condor_q"
echo ""

echo "2. 查看任务历史（如果已完成）："
echo "---"
echo "在CHTC上运行: condor_history -limit 5"
echo ""

echo "3. 检查结果目录："
echo "---"
echo "在CHTC上运行以下命令："
echo ""
echo "# 检查结果目录是否存在"
echo "ls -lh /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/ 2>/dev/null || echo '结果目录不存在'"
echo ""
echo "# 查看已处理的embryo数量"
echo "ls -1 /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/*.npy 2>/dev/null | wc -l"
echo ""
echo "# 查看metadata（包含处理统计）"
echo "cat /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/metadata.json 2>/dev/null || echo 'metadata不存在'"
echo ""
echo "# 查看最新的几个结果文件"
echo "ls -lht /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/*.npy 2>/dev/null | head -10"
echo ""

echo "4. 查看日志文件："
echo "---"
echo "在CHTC上运行："
echo ""
echo "# 查看输出日志（最后50行）"
echo "tail -50 ~/logs/extract_latents_v1_baseline.out"
echo ""
echo "# 查看错误日志"
echo "tail -50 ~/logs/extract_latents_v1_baseline.err"
echo ""
echo "# 查看condor日志（最后100行）"
echo "tail -100 ~/logs/extract_latents_v1_baseline.log"
echo ""

echo "5. 如果任务还在运行，查看实时输出："
echo "---"
echo "在CHTC上运行: condor_tail -f <job_id>"
echo ""

echo "=== 检查完成 ==="

