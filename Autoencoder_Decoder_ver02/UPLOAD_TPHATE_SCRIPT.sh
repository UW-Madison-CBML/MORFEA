#!/bin/bash
# 上传 tphate 生成脚本到 CHTC

echo "=== 上传 generate_tphate_for_aadhitya.py 到 CHTC ==="

SCRIPT_FILE="generate_tphate_for_aadhitya.py"
REMOTE_HOST="rho9@ap2001.chtc.wisc.edu"
REMOTE_DIR="/staging/groups/bhaskar_group/rho9"

if [ ! -f "$SCRIPT_FILE" ]; then
    echo "❌ 错误: $SCRIPT_FILE 不存在"
    exit 1
fi

echo "上传 $SCRIPT_FILE 到 $REMOTE_HOST:$REMOTE_DIR/"

scp "$SCRIPT_FILE" "$REMOTE_HOST:$REMOTE_DIR/"

if [ $? -eq 0 ]; then
    echo "✓ 上传成功！"
    echo ""
    echo "现在可以在 CHTC 上运行："
    echo "  ssh $REMOTE_HOST"
    echo "  cd $REMOTE_DIR"
    echo "  python3 generate_tphate_for_aadhitya.py --help"
else
    echo "❌ 上传失败"
    exit 1
fi






