#!/bin/bash
# upload_to_chtc_latents.sh
# 上传文件到CHTC用于提取latent trajectories

CHTC_USER="rho9"
CHTC_HOST="ap2001.chtc.wisc.edu"
CHTC_DIR="/staging/groups/bhaskar_group/rho9"

echo "=== 上传文件到CHTC ==="
echo "目标: ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}"
echo ""

# 确保在正确的目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "当前目录: $(pwd)"
echo ""

# 1. 上传Python脚本和配置文件
echo "1. 上传Python脚本和配置文件..."
scp extract_all_latent_trajectories.py \
    extract_latents.sh \
    extract_latents.sub \
    model.py \
    dataset_ivf.py \
    build_index.py \
    index.csv \
    ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/

if [ $? -eq 0 ]; then
    echo "✓ 文件上传成功"
else
    echo "✗ 文件上传失败"
    exit 1
fi

echo ""

# 2. 创建checkpoints目录（如果不存在）
echo "2. 确保checkpoints目录存在..."
ssh ${CHTC_USER}@${CHTC_HOST} "mkdir -p ${CHTC_DIR}/checkpoints"

# 3. 上传checkpoint
if [ -f "checkpoints/checkpoint_epoch_50.pt" ]; then
    echo "3. 上传checkpoint..."
    scp checkpoints/checkpoint_epoch_50.pt \
        ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/checkpoints/
    
    if [ $? -eq 0 ]; then
        echo "✓ Checkpoint上传成功"
    else
        echo "✗ Checkpoint上传失败"
        exit 1
    fi
else
    echo "⚠️  Warning: checkpoints/checkpoint_epoch_50.pt 不存在"
    echo "   请手动上传checkpoint文件"
fi

echo ""
echo "=== 上传完成 ==="
echo ""
echo "下一步："
echo "1. SSH到CHTC: ssh ${CHTC_USER}@${CHTC_HOST}"
echo "2. 进入目录: cd ${CHTC_DIR}"
echo "3. 编辑submit文件: nano extract_latents.sub"
echo "4. 添加以下内容到文件末尾："
echo "   checkpoint = checkpoints/checkpoint_epoch_50.pt"
echo "   model_version = v1_baseline"
echo "   queue"
echo "5. 创建logs目录: mkdir -p logs"
echo "6. 提交任务: condor_submit extract_latents.sub"

