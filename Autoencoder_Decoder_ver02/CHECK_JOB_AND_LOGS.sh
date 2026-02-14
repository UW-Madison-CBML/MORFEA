#!/bin/bash
# 检查作业状态和日志文件

echo "=== 检查作业状态 ==="
condor_q -submitter rho9
echo ""

echo "=== 查找日志文件 ==="
echo "1. 在 home 目录查找:"
find ~/logs -name "generate_tphate_*" -type f 2>/dev/null | head -5
echo ""

echo "2. 检查 logs 目录是否存在:"
ls -la ~/logs/ 2>/dev/null || echo "  ⚠️  ~/logs/ 目录不存在"
echo ""

echo "3. 检查所有可能的日志位置:"
find ~ -name "*generate_tphate*" -type f 2>/dev/null | head -10
echo ""

echo "4. 检查作业的详细信息:"
condor_q -submitter rho9 -format "Cluster: %d\n" Cluster -format "Process: %d\n" Process -format "Status: %s\n" JobStatus -format "Log: %s\n" UserLog 2>/dev/null | head -20
echo ""

echo "=== 如果作业在运行但日志不存在 ==="
echo "可能原因："
echo "  1. 作业刚启动，还没开始写日志"
echo "  2. 日志路径配置有问题"
echo ""
echo "建议："
echo "  - 等待几分钟后再检查"
echo "  - 或者直接检查输出目录的进度"






