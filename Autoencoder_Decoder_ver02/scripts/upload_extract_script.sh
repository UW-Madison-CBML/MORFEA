#!/bin/bash
# 上傳解壓腳本到 CHTC

echo "============================================================"
echo "上傳解壓腳本到 CHTC"
echo "============================================================"
echo ""

SCRIPT_LOCAL="scripts/extract_from_group_tar.sh"
SCRIPT_REMOTE="rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/ivf_analysis/scripts/extract_from_group_tar.sh"

if [ ! -f "$SCRIPT_LOCAL" ]; then
    echo "❌ 錯誤: 本地腳本不存在: $SCRIPT_LOCAL"
    exit 1
fi

echo "上傳: $SCRIPT_LOCAL"
echo "到: $SCRIPT_REMOTE"
echo ""

scp "$SCRIPT_LOCAL" "$SCRIPT_REMOTE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 上傳成功！"
    echo ""
    echo "現在可以在 CHTC 上執行:"
    echo "  cd /staging/groups/bhaskar_group/rho9/ivf_analysis"
    echo "  bash scripts/extract_from_group_tar.sh"
else
    echo ""
    echo "❌ 上傳失敗，請檢查 SSH 連接"
    exit 1
fi

