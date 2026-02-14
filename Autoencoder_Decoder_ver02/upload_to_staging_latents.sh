#!/bin/bash
# upload_to_staging_latents.sh
# 上传文件到CHTC staging目录用于提取latent trajectories

CHTC_USER="rho9"
CHTC_HOST="ap2001.chtc.wisc.edu"
STAGING_DIR="/staging/groups/bhaskar_group/rho9"

echo "=== 上传文件到CHTC Staging ==="
echo "目标: ${CHTC_USER}@${CHTC_HOST}:${STAGING_DIR}"
echo ""

# 确保在正确的目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "当前目录: $(pwd)"
echo ""

# 1. 上传Python脚本和配置文件到staging
echo "1. 上传Python脚本和配置文件到staging..."
scp extract_all_latent_trajectories.py \
    extract_latents.sh \
    extract_latents.sub \
    model.py \
    dataset_ivf.py \
    build_index.py \
    index.csv \
    ${CHTC_USER}@${CHTC_HOST}:${STAGING_DIR}/

if [ $? -eq 0 ]; then
    echo "✓ 文件上传成功到staging"
else
    echo "✗ 文件上传失败"
    exit 1
fi

echo ""

# 2. 确保checkpoints目录存在（在staging）
echo "2. 确保checkpoints目录存在（在staging）..."
ssh ${CHTC_USER}@${CHTC_HOST} "mkdir -p ${STAGING_DIR}/checkpoints"

# 3. 上传checkpoint到staging
if [ -f "checkpoints/checkpoint_epoch_50.pt" ]; then
    echo "3. 上传checkpoint到staging..."
    scp checkpoints/checkpoint_epoch_50.pt \
        ${CHTC_USER}@${CHTC_HOST}:${STAGING_DIR}/checkpoints/
    
    if [ $? -eq 0 ]; then
        echo "✓ Checkpoint上传成功到staging"
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
echo "文件已上传到staging目录: ${STAGING_DIR}"
echo ""
echo "下一步："
echo "1. SSH到CHTC: ssh ${CHTC_USER}@${CHTC_HOST}"
echo "2. 进入staging目录: cd ${STAGING_DIR}"
echo "3. 编辑submit文件: nano extract_latents.sub"
echo "4. 添加以下内容到文件末尾："
echo "   checkpoint = checkpoints/checkpoint_epoch_50.pt"
echo "   model_version = v1_baseline"
echo "   queue"
echo "5. 创建logs目录: mkdir -p logs"
echo "6. 提交任务: condor_submit extract_latents.sub"
echo ""
echo "注意：checkpoint路径在submit文件中应该是相对路径，"
echo "     因为文件已经在staging目录中了"

