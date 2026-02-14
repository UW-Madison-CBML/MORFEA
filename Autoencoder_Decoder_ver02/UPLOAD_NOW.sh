#!/bin/bash
# 在本地執行：上傳修復後的文件到 CHTC

REMOTE_USER="rho9"
REMOTE_HOST="ap2001.chtc.wisc.edu"
STAGING_DIR="/staging/groups/bhaskar_group/rho9"

echo "============================================================"
echo "上傳修復後的文件到 CHTC（從本地）"
echo "============================================================"
echo ""

cd "$(dirname "$0")"
echo "當前目錄: $(pwd)"
echo ""

# 檢查文件是否存在
if [ ! -f "extract_latents.sh" ]; then
    echo "❌ 錯誤: extract_latents.sh 不存在於當前目錄"
    exit 1
fi

if [ ! -f "extract_latents_from_home.sub" ]; then
    echo "❌ 錯誤: extract_latents_from_home.sub 不存在於當前目錄"
    exit 1
fi

echo "要上傳的文件："
ls -lh extract_latents.sh extract_latents_from_home.sub
echo ""

echo "1️⃣  上傳 extract_latents.sh 到 staging..."
scp extract_latents.sh ${REMOTE_USER}@${REMOTE_HOST}:${STAGING_DIR}/

if [ $? -eq 0 ]; then
    echo "   ✅ extract_latents.sh 上傳成功"
else
    echo "   ❌ 上傳失敗，請檢查網絡連接"
    exit 1
fi

echo ""
echo "2️⃣  上傳 extract_latents_from_home.sub 到 home 目錄..."
scp extract_latents_from_home.sub ${REMOTE_USER}@${REMOTE_HOST}:~/

if [ $? -eq 0 ]; then
    echo "   ✅ extract_latents_from_home.sub 上傳成功"
else
    echo "   ❌ 上傳失敗，請檢查網絡連接"
    exit 1
fi

echo ""
echo "============================================================"
echo "✅ 上傳完成！"
echo "============================================================"
echo ""
echo "現在請在 CHTC 上執行以下命令："
echo ""
echo "   ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo ""
echo "然後："
echo "   1. 移除舊任務: condor_rm 2851275.0"
echo "   2. 檢查文件: ls -lh ~/extract_latents_from_home.sub"
echo "   3. 檢查 staging: ls -lh ${STAGING_DIR}/extract_latents.sh"
echo "   4. 檢查並設置 submit 文件: tail -10 ~/extract_latents_from_home.sub"
echo "   5. 提交任務: condor_submit ~/extract_latents_from_home.sub"
echo ""
echo "============================================================"

