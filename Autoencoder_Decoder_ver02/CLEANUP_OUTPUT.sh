#!/bin/bash
# 清理输出目录

echo "=== 停止正在运行的进程 ==="
pkill -f "generate_tphate_for_aadhitya.py" 2>/dev/null
sleep 2
echo "✓ 进程已停止"

echo ""
echo "=== 删除输出目录 ==="

# 删除 rho9 目录的输出
if [ -d "/staging/groups/bhaskar_group/rho9/aadhitya_v1_val" ]; then
    echo "删除: /staging/groups/bhaskar_group/rho9/aadhitya_v1_val"
    rm -rf /staging/groups/bhaskar_group/rho9/aadhitya_v1_val
    echo "✓ 已删除"
fi

# 删除 ivf 目录的输出
if [ -d "/staging/groups/bhaskar_group/ivf/aadhitya_v1_val" ]; then
    echo "删除: /staging/groups/bhaskar_group/ivf/aadhitya_v1_val"
    rm -rf /staging/groups/bhaskar_group/ivf/aadhitya_v1_val
    echo "✓ 已删除"
fi

echo ""
echo "=== 清理完成 ==="
echo "现在可以重新开始运行"






