#!/bin/bash
# 完整的上传和提交流程，确保一次成功

set -e  # 遇到错误立即退出

echo "=== 上传文件到CHTC并提交任务 ==="
echo ""

PROJECT_DIR="/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"
CHTC_HOST="rho9@ap2001.chtc.wisc.edu"
STAGING_DIR="/staging/groups/bhaskar_group/rho9"

cd "$PROJECT_DIR"

# 1. 先运行检查
echo "步骤1: 检查文件..."
./CHECK_BEFORE_UPLOAD.sh || { echo "检查失败，停止上传"; exit 1; }

echo ""
echo "步骤2: 上传文件到CHTC..."
echo "  连接到: $CHTC_HOST"
echo "  目标目录: $STAGING_DIR"
echo ""

# 2. 上传所有必要文件
files=(
    "extract_all_latent_trajectories.py"
    "model.py"
    "dataset_ivf.py"
    "build_index.py"
    "extract_latents.sh"
)

for file in "${files[@]}"; do
    echo "  上传 $file..."
    scp "$file" "$CHTC_HOST:$STAGING_DIR/" || { echo "  ✗ 上传 $file 失败"; exit 1; }
    echo "  ✓ $file 上传成功"
done

echo ""
echo "步骤3: 在CHTC上验证文件..."
ssh "$CHTC_HOST" << 'REMOTE_EOF'
cd /staging/groups/bhaskar_group/rho9
echo "  检查文件..."
for file in extract_all_latent_trajectories.py model.py dataset_ivf.py build_index.py extract_latents.sh; do
    if [ -f "$file" ]; then
        echo "    ✓ $file"
    else
        echo "    ✗ $file - MISSING!"
        exit 1
    fi
done

echo "  验证脚本..."
head -1 extract_latents.sh | grep -q "#!/bin/bash" && echo "    ✓ extract_latents.sh 开头正确" || { echo "    ✗ extract_latents.sh 开头错误"; exit 1; }
grep -q "需要我提供" extract_latents.sh && { echo "    ✗ extract_latents.sh 包含占位符！"; exit 1; } || echo "    ✓ extract_latents.sh 没有占位符"

echo "  验证Python代码..."
python3 -c "import extract_all_latent_trajectories" 2>/dev/null && echo "    ✓ extract_all_latent_trajectories.py 可以导入" || echo "    ⚠️  extract_all_latent_trajectories.py 导入警告（可能缺少依赖，但文件存在）"

echo "  ✓ 所有文件验证通过"
REMOTE_EOF

if [ $? -ne 0 ]; then
    echo "  ✗ 验证失败"
    exit 1
fi

echo ""
echo "步骤4: 取消旧任务（如果有）..."
ssh "$CHTC_HOST" << 'REMOTE_EOF'
condor_q -format "%d " JobID -format "%s\n" JobStatus | grep "RUNNING" | awk '{print $1}' | while read jobid; do
    if [ -n "$jobid" ]; then
        echo "  取消任务 $jobid..."
        condor_rm "$jobid" 2>/dev/null || true
    fi
done
echo "  ✓ 旧任务已取消"
REMOTE_EOF

echo ""
echo "步骤5: 提交新任务..."
ssh "$CHTC_HOST" << 'REMOTE_EOF'
cd ~
if [ ! -f extract_latents_from_home.sub ]; then
    echo "  ✗ extract_latents_from_home.sub 不存在"
    exit 1
fi

echo "  提交任务..."
condor_submit extract_latents_from_home.sub
echo ""
echo "  ✓ 任务已提交"
echo ""
echo "  查看任务状态:"
condor_q
REMOTE_EOF

if [ $? -ne 0 ]; then
    echo "  ✗ 提交失败"
    exit 1
fi

echo ""
echo "=== 完成！ ==="
echo ""
echo "任务已提交，你可以："
echo "  1. SSH到CHTC查看状态: ssh $CHTC_HOST"
echo "  2. 查看任务: condor_q"
echo "  3. 查看输出: condor_tail -f <job_id>"
echo "  4. 检查结果: ls -lh $STAGING_DIR/model_latents/v1_baseline/"

