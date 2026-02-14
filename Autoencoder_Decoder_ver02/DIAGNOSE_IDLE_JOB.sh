#!/bin/bash
# 诊断为什么作业是 idle 状态

echo "=== 检查作业详细信息 ==="
echo ""

# 1. 查看作业状态和原因
condor_q -submitter rho9 -better-analyze
echo ""

# 2. 查看作业的完整信息
echo "=== 作业详细信息 ==="
condor_q -submitter rho9 -format "Cluster: %d\n" Cluster \
         -format "Process: %d\n" Process \
         -format "Status: %s\n" JobStatus \
         -format "RequestCPUs: %s\n" RequestCpus \
         -format "RequestMemory: %s\n" RequestMemory \
         -format "RequestDisk: %s\n" RequestDisk \
         -format "HoldReason: %s\n" HoldReason \
         -format "LastHoldReason: %s\n" LastHoldReason 2>/dev/null
echo ""

# 3. 检查是否有 hold 的作业
echo "=== 检查是否有 hold 的作业 ==="
condor_q -submitter rho9 -hold
echo ""

# 4. 检查输出目录进度
echo "=== 当前进度 ==="
OUTPUT_DIR="/staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
if [ -d "$OUTPUT_DIR/tphate_plots" ]; then
    COUNT=$(ls -1 "$OUTPUT_DIR/tphate_plots"/*.png 2>/dev/null | wc -l)
    echo "已生成: $COUNT / 704 T-PHATE plots"
    echo "剩余: $((704 - COUNT)) 个胚胎"
fi

echo ""
echo "如果作业一直是 idle，可能原因："
echo "  1. 资源不足（CPU/内存）"
echo "  2. 作业被 hold"
echo "  3. 没有可用的计算节点"
echo ""
echo "建议："
echo "  - 运行 condor_q -better-analyze 查看详细原因"
echo "  - 或者降低资源要求（减少 request_cpus 或 request_memory）"






