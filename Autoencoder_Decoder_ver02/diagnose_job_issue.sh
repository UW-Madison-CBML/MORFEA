#!/bin/bash
# diagnose_job_issue.sh
# 诊断任务为什么消失

echo "=== 诊断任务问题 ==="
echo ""

# 1. 检查任务历史
echo "1. 检查最近的任务历史："
condor_history -limit 5 -format "%d\n" ClusterId -format "  Job %s: " ClusterId -format "Status=%s " JobStatus -format "ExitCode=%s\n" ExitCode 2>/dev/null || echo "  无法获取历史"

echo ""
echo "2. 检查日志文件："
if [ -f "logs/extract_latents_v1_baseline.out" ]; then
    echo "  输出日志存在，最后20行："
    tail -20 logs/extract_latents_v1_baseline.out
else
    echo "  ⚠️  输出日志不存在"
fi

echo ""
if [ -f "logs/extract_latents_v1_baseline.err" ]; then
    echo "  错误日志存在，内容："
    cat logs/extract_latents_v1_baseline.err
else
    echo "  ⚠️  错误日志不存在"
fi

echo ""
if [ -f "logs/extract_latents_v1_baseline.log" ]; then
    echo "  Condor日志存在，最后30行："
    tail -30 logs/extract_latents_v1_baseline.log
else
    echo "  ⚠️  Condor日志不存在"
fi

echo ""
echo "3. 检查submit文件语法："
condor_submit -dry-run extract_latents_from_home.sub 2>&1 | head -20

echo ""
echo "4. 检查GPU机器可用性："
condor_status -const 'Machine == "bhaskargpu4000.chtc.wisc.edu"' -af Name GPUs State Activity 2>/dev/null || echo "  ⚠️  无法检查bhaskargpu4000状态"

echo ""
echo "5. 检查所有bhaskar GPU机器："
condor_status -constraint 'regexp("bhaskar", Machine)' -af Name GPUs State Activity 2>/dev/null | head -10

echo ""
echo "6. 检查是否有hold的任务："
condor_q -hold

echo ""
echo "=== 诊断完成 ==="

