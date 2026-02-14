#!/bin/bash
# 從 group 層級的 tar.gz 解壓特定的 cells

set -e

echo "============================================================"
echo "從 Group Tar.gz 解壓特定 Cells"
echo "============================================================"
echo ""

# 定義路徑
GROUP_TAR_FILE="/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz"
STAGING_DIR="/staging/groups/bhaskar_group/rho9/ivf_data"
EXTRACT_DIR="$STAGING_DIR/embryo_dataset"

# 預設要解壓的 cells (根據分析需求)
DEFAULT_CELLS=("ZS435-5" "RS363-7")

# 檢查 tar.gz 是否存在
if [ ! -f "$GROUP_TAR_FILE" ]; then
    echo "❌ 錯誤: Group tar.gz 檔案不存在:"
    echo "   $GROUP_TAR_FILE"
    exit 1
fi

TAR_SIZE=$(du -sh "$GROUP_TAR_FILE" | cut -f1)
TAR_OWNER=$(ls -l "$GROUP_TAR_FILE" | awk '{print $3}')
echo "✓ 找到 Group tar.gz 檔案:"
echo "   路徑: $GROUP_TAR_FILE"
echo "   大小: $TAR_SIZE"
echo "   擁有者: $TAR_OWNER"
echo ""

# 如果沒有提供參數，使用預設的 cells
if [ $# -eq 0 ]; then
    echo "沒有指定 cells，使用預設的 cells:"
    CELLS=("${DEFAULT_CELLS[@]}")
else
    CELLS=("$@")
fi

echo "要解壓的 cells:"
for cell in "${CELLS[@]}"; do
    echo "  - $cell"
done
echo ""

# 檢查是否已經有解壓目錄
if [ -d "$EXTRACT_DIR" ]; then
    EXTRACT_COUNT=$(ls -d "$EXTRACT_DIR"/*/ 2>/dev/null | wc -l)
    EXTRACT_SIZE=$(du -sh "$EXTRACT_DIR" 2>/dev/null | cut -f1)
    echo "⚠️  已存在解壓目錄: $EXTRACT_DIR"
    echo "   目前有 $EXTRACT_COUNT 個 cells, 總大小: $EXTRACT_SIZE"
    echo ""
    echo "選項:"
    echo "  1. 保留現有目錄，只解壓缺少的 cells (推薦)"
    echo "  2. 刪除現有目錄並重新解壓"
    echo ""
    read -p "選擇 (1/2，預設 1): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[2]$ ]]; then
        echo "刪除現有目錄..."
        rm -rf "$EXTRACT_DIR"
        echo "✓ 已刪除"
        echo ""
    else
        echo "保留現有目錄，只解壓缺少的 cells"
        echo ""
    fi
fi

# 確保目標目錄存在
mkdir -p "$EXTRACT_DIR"

# 檢查 staging 空間
echo "============================================================"
echo "檢查 Staging 空間"
echo "============================================================"
echo ""
STAGING_AVAIL=$(df -h "$STAGING_DIR" | tail -1 | awk '{print $4}')
echo "可用空間: $STAGING_AVAIL"
echo ""

# 解壓特定 cells
echo "============================================================"
echo "開始解壓..."
echo "============================================================"
echo ""

# 設定 TMPDIR 到 staging，避免使用 home directory
export TMPDIR="/staging/groups/bhaskar_group/rho9/tmp"
mkdir -p "$TMPDIR"

cd "$STAGING_DIR"

for cell in "${CELLS[@]}"; do
    echo "解壓: $cell"
    
    # 檢查 cell 是否已經存在
    if [ -d "$EXTRACT_DIR/$cell" ]; then
        CELL_SIZE=$(du -sh "$EXTRACT_DIR/$cell" 2>/dev/null | cut -f1)
        FRAME_COUNT=$(ls "$EXTRACT_DIR/$cell"/*.jpeg "$EXTRACT_DIR/$cell"/*.jpg 2>/dev/null | wc -l)
        echo "  ⚠️  已存在: $CELL_SIZE, $FRAME_COUNT frames"
        echo "  跳過解壓"
        continue
    fi
    
    # 從 tar.gz 解壓特定 cell
    echo "  正在解壓..."
    tar -xzf "$GROUP_TAR_FILE" -C "$STAGING_DIR" "embryo_dataset/$cell" 2>&1 | head -10
    
    if [ $? -eq 0 ]; then
        if [ -d "$EXTRACT_DIR/$cell" ]; then
            CELL_SIZE=$(du -sh "$EXTRACT_DIR/$cell" 2>/dev/null | cut -f1)
            FRAME_COUNT=$(ls "$EXTRACT_DIR/$cell"/*.jpeg "$EXTRACT_DIR/$cell"/*.jpg 2>/dev/null | wc -l)
            echo "  ✓ 完成: $CELL_SIZE, $FRAME_COUNT frames"
        else
            echo "  ⚠️  解壓完成但目錄不存在，可能 tar.gz 中沒有這個 cell"
            echo "  檢查 tar.gz 內容..."
            tar -tzf "$GROUP_TAR_FILE" | grep "^embryo_dataset/$cell/" | head -5
        fi
    else
        echo "  ❌ 解壓失敗"
    fi
    echo ""
done

# 驗證結果
echo "============================================================"
echo "驗證結果"
echo "============================================================"
echo ""

if [ -d "$EXTRACT_DIR" ]; then
    FINAL_SIZE=$(du -sh "$EXTRACT_DIR" 2>/dev/null | cut -f1)
    FINAL_COUNT=$(ls -d "$EXTRACT_DIR"/*/ 2>/dev/null | wc -l)
    echo "解壓目錄: $EXTRACT_DIR"
    echo "總大小: $FINAL_SIZE"
    echo "Cell 目錄數量: $FINAL_COUNT"
    echo ""
    
    echo "已解壓的 cells:"
    ls -d "$EXTRACT_DIR"/*/ 2>/dev/null | xargs -n1 basename
    echo ""
    
    # 檢查是否所有需要的 cells 都存在
    echo "檢查需要的 cells:"
    ALL_EXIST=true
    for cell in "${CELLS[@]}"; do
        if [ -d "$EXTRACT_DIR/$cell" ]; then
            CELL_SIZE=$(du -sh "$EXTRACT_DIR/$cell" 2>/dev/null | cut -f1)
            FRAME_COUNT=$(ls "$EXTRACT_DIR/$cell"/*.jpeg "$EXTRACT_DIR/$cell"/*.jpg 2>/dev/null | wc -l)
            echo "  ✓ $cell ($CELL_SIZE, $FRAME_COUNT frames)"
        else
            echo "  ❌ $cell (不存在)"
            ALL_EXIST=false
        fi
    done
    echo ""
    
    if [ "$ALL_EXIST" = true ]; then
        echo "✅ 所有需要的 cells 都已解壓！"
    else
        echo "⚠️  部分 cells 不存在，請檢查 tar.gz 內容"
    fi
else
    echo "❌ 錯誤: 解壓目錄不存在"
    exit 1
fi

echo ""
echo "============================================================"
echo "使用方式"
echo "============================================================"
echo ""
echo "現在你可以使用解壓的 cells:"
echo "  data_root: $EXTRACT_DIR"
echo ""
echo "例如，在分析腳本中使用:"
echo "  --data_root $EXTRACT_DIR"
echo ""
echo "或者讓腳本自動偵測（腳本會自動尋找這個路徑）"
echo ""

