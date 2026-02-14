#!/bin/bash
# 從 tar.gz 只解壓特定的 cells，節省空間

set -e

echo "============================================================"
echo "從 tar.gz 解壓特定 Cells"
echo "============================================================"
echo ""

# 定義路徑
TAR_FILE="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"
STAGING_DIR="/staging/groups/bhaskar_group/rho9/ivf_data"
EXTRACT_DIR="$STAGING_DIR/embryo_dataset"

# 預設要解壓的 cells (根據你的分析需求)
DEFAULT_CELLS=("RI382-2" "ZS435-5" "RS363-7")

# 檢查 tar.gz 是否存在
if [ ! -f "$TAR_FILE" ]; then
    echo "❌ 錯誤: tar.gz 檔案不存在:"
    echo "   $TAR_FILE"
    exit 1
fi

TAR_SIZE=$(du -sh "$TAR_FILE" | cut -f1)
echo "✓ 找到 tar.gz 檔案: $TAR_SIZE"
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
    echo "⚠️  已存在解壓目錄: $EXTRACT_DIR"
    read -p "要刪除現有的並重新解壓嗎? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
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

# 解壓特定 cells
echo "============================================================"
echo "開始解壓..."
echo "============================================================"
echo ""

cd "$STAGING_DIR"

for cell in "${CELLS[@]}"; do
    echo "解壓: $cell"
    
    # 檢查 cell 是否已經存在
    if [ -d "$EXTRACT_DIR/$cell" ]; then
        echo "  ⚠️  已存在，跳過"
        continue
    fi
    
    # 從 tar.gz 解壓特定 cell
    tar -xzvf "$TAR_FILE" "embryo_dataset/$cell" 2>&1 | head -5
    
    if [ $? -eq 0 ]; then
        if [ -d "$EXTRACT_DIR/$cell" ]; then
            CELL_SIZE=$(du -sh "$EXTRACT_DIR/$cell" 2>/dev/null | cut -f1)
            FRAME_COUNT=$(ls "$EXTRACT_DIR/$cell"/*.jpeg 2>/dev/null | wc -l)
            echo "  ✓ 完成: $CELL_SIZE, $FRAME_COUNT frames"
        else
            echo "  ⚠️  解壓完成但目錄不存在，可能 tar.gz 中沒有這個 cell"
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
    for cell in "${CELLS[@]}"; do
        if [ -d "$EXTRACT_DIR/$cell" ]; then
            echo "  ✓ $cell"
        else
            echo "  ❌ $cell (不存在)"
        fi
    done
    echo ""
    
    echo "✅ 完成！"
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

