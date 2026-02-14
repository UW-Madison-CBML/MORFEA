#!/bin/bash
# 快速更新CHTC上的脚本文件

CHTC_USER="rho9"
CHTC_HOST="ap2001.chtc.wisc.edu"
CHTC_DIR="/staging/groups/bhaskar_group/rho9"

echo "=== 快速更新CHTC脚本 ==="
echo ""

# 确保在正确的目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "上传修复后的文件..."
scp extract_latents.sh \
    extract_all_latent_trajectories.py \
    ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 文件上传成功"
    echo ""
    echo "下一步："
    echo "1. SSH到CHTC: ssh ${CHTC_USER}@${CHTC_HOST}"
    echo "2. 取消旧任务: condor_rm 2851098.0"
    echo "3. 确认脚本已更新: head -5 ${CHTC_DIR}/extract_latents.sh"
    echo "4. 重新提交: cd ~ && condor_submit extract_latents_from_home.sub"
else
    echo "✗ 上传失败"
    exit 1
fi

