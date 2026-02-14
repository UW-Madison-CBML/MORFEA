#!/bin/bash
# 检查 GPU 作业为什么是 idle

echo "=== 检查 GPU 作业详细信息 ==="
condor_q -submitter rho9 -better-analyze
echo ""

echo "=== 检查 GPU 节点可用性 ==="
condor_status -constraint 'GPUs > 0' -format "Name: %s\n" Name -format "GPUs: %d\n" GPUs -format "State: %s\n" State | head -20
echo ""

echo "=== 检查 bhaskar GPU 节点 ==="
condor_status -constraint 'Machine == "bhaskargpu4000.chtc.wisc.edu"' -format "Name: %s\n" Name -format "State: %s\n" State -format "GPUs: %d\n" GPUs 2>/dev/null || echo "节点不存在或不可访问"
echo ""

echo "=== 建议 ==="
echo "如果 GPU 节点不可用，可以："
echo "1. 降低资源要求（减少 request_cpus 或 request_memory）"
echo "2. 移除 GPU 要求，使用普通 CPU 节点"
echo "3. 等待 GPU 节点可用"






