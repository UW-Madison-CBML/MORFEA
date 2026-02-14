#!/bin/bash
# 上傳修復後的文件並重新提交任務

REMOTE_USER="rho9"
REMOTE_HOST="ap2001.chtc.wisc.edu"
STAGING_DIR="/staging/groups/bhaskar_group/rho9"

echo "============================================================"
echo "上傳修復後的文件到 CHTC"
echo "============================================================"
echo ""

cd "$(dirname "$0")"

# 檢查文件是否存在
if [ ! -f "extract_latents.sh" ]; then
    echo "❌ 錯誤: extract_latents.sh 不存在"
    exit 1
fi

if [ ! -f "extract_latents_from_home.sub" ]; then
    echo "❌ 錯誤: extract_latents_from_home.sub 不存在"
    exit 1
fi

echo "1. 上傳修復後的 extract_latents.sh 到 staging..."
scp extract_latents.sh ${REMOTE_USER}@${REMOTE_HOST}:${STAGING_DIR}/

if [ $? -eq 0 ]; then
    echo "   ✓ extract_latents.sh 上傳成功"
else
    echo "   ❌ 上傳失敗"
    exit 1
fi

echo ""
echo "2. 上傳修復後的 extract_latents_from_home.sub 到 home 目錄..."
scp extract_latents_from_home.sub ${REMOTE_USER}@${REMOTE_HOST}:~/

if [ $? -eq 0 ]; then
    echo "   ✓ extract_latents_from_home.sub 上傳成功"
else
    echo "   ❌ 上傳失敗"
    exit 1
fi

echo ""
echo "============================================================"
echo "下一步操作："
echo ""
echo "1. SSH 到 CHTC："
echo "   ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo ""
echo "2. 移除舊任務（如果還在）："
echo "   condor_rm 2851275.0"
echo ""
echo "3. 確認 submit 文件設置正確（檢查最後幾行）："
echo "   tail -10 ~/extract_latents_from_home.sub"
echo ""
echo "   應該看到類似："
echo "   checkpoint = /staging/groups/bhaskar_group/rho9/checkpoints/checkpoint_epoch_50.pt"
echo "   model_version = v1_baseline"
echo "   queue"
echo ""
echo "4. 如果沒有設置，添加並提交："
echo "   echo 'checkpoint = /staging/groups/bhaskar_group/rho9/checkpoints/checkpoint_epoch_50.pt' >> ~/extract_latents_from_home.sub"
echo "   echo 'model_version = v1_baseline' >> ~/extract_latents_from_home.sub"
echo "   echo 'queue' >> ~/extract_latents_from_home.sub"
echo "   condor_submit ~/extract_latents_from_home.sub"
echo ""
echo "5. 監控任務："
echo "   condor_q -submitter rho9"
echo ""
echo "6. 實時查看輸出（當任務開始運行後）："
echo "   condor_tail -f <job_id>"
echo ""
echo "============================================================"

