#!/bin/bash
# 檢查 tar.gz 內部的檔案結構

set -e

TAR_FILE="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"

echo "============================================================"
echo "檢查 tar.gz 內部結構"
echo "============================================================"
echo ""

if [ ! -f "$TAR_FILE" ]; then
    echo "❌ tar.gz 檔案不存在: $TAR_FILE"
    exit 1
fi

# 檢查特定 cells 的結構
CELLS=("RI382-2" "ZS435-5" "RS363-7")

for cell in "${CELLS[@]}"; do
    echo "============================================================"
    echo "檢查 $cell 的結構:"
    echo "============================================================"
    
    # 列出這個 cell 的所有檔案路徑
    echo "所有檔案路徑:"
    tar -tzf "$TAR_FILE" | grep "^embryo_dataset/$cell" | head -20
    echo ""
    
    # 計算檔案數量
    FILE_COUNT=$(tar -tzf "$TAR_FILE" | grep "^embryo_dataset/$cell" | wc -l)
    echo "總檔案數: $FILE_COUNT"
    echo ""
    
    # 檢查是否有子目錄
    DIRS=$(tar -tzf "$TAR_FILE" | grep "^embryo_dataset/$cell" | grep "/" | cut -d'/' -f3 | sort -u)
    if [ -n "$DIRS" ]; then
        echo "子目錄:"
        echo "$DIRS" | grep -v "^$cell$"
        echo ""
    fi
    echo ""
done

echo "============================================================"
echo "建議"
echo "============================================================"
echo ""
echo "如果檔案在子目錄中（例如 F0/），需要解壓完整路徑"
echo "例如: tar -xzvf embryo_dataset.tar.gz 'embryo_dataset/RI382-2/F0/*'"
echo ""

