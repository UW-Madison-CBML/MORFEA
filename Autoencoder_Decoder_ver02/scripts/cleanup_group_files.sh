#!/bin/bash
# 檢查並清理 group staging 目錄，找出可以刪除的檔案

set -e

echo "============================================================"
echo "Group Staging 目錄清理檢查"
echo "============================================================"
echo ""

STAGING_BASE="/staging/groups/bhaskar_group/rho9"

echo "檢查目錄: $STAGING_BASE"
echo ""

# 1. 列出所有目錄和檔案大小
echo "============================================================"
echo "1. 所有目錄和檔案大小"
echo "============================================================"
if [ -d "$STAGING_BASE" ]; then
    echo "總大小:"
    du -sh "$STAGING_BASE"
    echo ""
    echo "各目錄詳細大小:"
    du -sh "$STAGING_BASE"/* 2>/dev/null | sort -h
    echo ""
else
    echo "❌ 目錄不存在: $STAGING_BASE"
    exit 1
fi

# 2. 檢查 ivf_data 目錄
echo "============================================================"
echo "2. IVF Data 目錄詳細檢查"
echo "============================================================"
IVF_DATA_DIR="$STAGING_BASE/ivf_data"

if [ -d "$IVF_DATA_DIR" ]; then
    echo "位置: $IVF_DATA_DIR"
    echo ""
    
    echo "所有檔案和目錄:"
    ls -lh "$IVF_DATA_DIR" 2>/dev/null
    echo ""
    
    # 檢查 tar.gz
    TAR_FILE="$IVF_DATA_DIR/embryo_dataset.tar.gz"
    if [ -f "$TAR_FILE" ]; then
        TAR_SIZE=$(du -sh "$TAR_FILE" | cut -f1)
        TAR_SIZE_BYTES=$(du -sb "$TAR_FILE" | cut -f1)
        TAR_SIZE_GB=$((TAR_SIZE_BYTES / 1024 / 1024 / 1024))
        echo "✓ 找到 tar.gz 檔案:"
        echo "   大小: $TAR_SIZE (~${TAR_SIZE_GB}GB)"
        echo "   這是壓縮檔，解壓後會是 ~12GB"
        echo ""
    fi
    
    # 檢查已解壓的目錄
    EXTRACT_DIR="$IVF_DATA_DIR/embryo_dataset"
    if [ -d "$EXTRACT_DIR" ]; then
        EXTRACT_SIZE=$(du -sh "$EXTRACT_DIR" 2>/dev/null | cut -f1)
        EXTRACT_COUNT=$(ls -d "$EXTRACT_DIR"/*/ 2>/dev/null | wc -l)
        echo "✓ 找到已解壓目錄:"
        echo "   大小: $EXTRACT_SIZE"
        echo "   Cell 目錄數量: $EXTRACT_COUNT"
        echo ""
        
        if [ "$EXTRACT_COUNT" -lt 10 ]; then
            echo "   ⚠️  這個解壓縮不完整！可以刪除"
            echo "   刪除命令: rm -rf $EXTRACT_DIR"
            echo ""
        fi
    else
        echo "⚠️  沒有已解壓的目錄"
        echo ""
    fi
else
    echo "❌ IVF data 目錄不存在"
fi

# 3. 檢查 ivf_analysis 目錄
echo "============================================================"
echo "3. IVF Analysis 目錄檢查"
echo "============================================================"
IVF_ANALYSIS_DIR="$STAGING_BASE/ivf_analysis"

if [ -d "$IVF_ANALYSIS_DIR" ]; then
    echo "位置: $IVF_ANALYSIS_DIR"
    ANALYSIS_SIZE=$(du -sh "$IVF_ANALYSIS_DIR" 2>/dev/null | cut -f1)
    echo "總大小: $ANALYSIS_SIZE"
    echo ""
    
    echo "子目錄:"
    du -sh "$IVF_ANALYSIS_DIR"/* 2>/dev/null | sort -h
    echo ""
    
    # 檢查 __pycache__ (可以刪除)
    if [ -d "$IVF_ANALYSIS_DIR/__pycache__" ]; then
        CACHE_SIZE=$(du -sh "$IVF_ANALYSIS_DIR/__pycache__" 2>/dev/null | cut -f1)
        echo "⚠️  找到 __pycache__ (可以刪除): $CACHE_SIZE"
        echo "   刪除命令: rm -rf $IVF_ANALYSIS_DIR/__pycache__"
        echo ""
    fi
else
    echo "⚠️  Analysis 目錄不存在"
fi

# 4. 檢查 curvature_analysis 目錄
echo "============================================================"
echo "4. Curvature Analysis 目錄檢查"
echo "============================================================"
CURV_DIR="$STAGING_BASE/curvature_analysis"

if [ -d "$CURV_DIR" ]; then
    CURV_SIZE=$(du -sh "$CURV_DIR" 2>/dev/null | cut -f1)
    echo "位置: $CURV_DIR"
    echo "大小: $CURV_SIZE"
    echo ""
    
    echo "內容:"
    ls -lh "$CURV_DIR" 2>/dev/null | head -20
    echo ""
else
    echo "⚠️  Curvature analysis 目錄不存在"
fi

# 5. 總結和建議
echo "============================================================"
echo "5. 清理建議"
echo "============================================================"
echo ""

# 可以刪除的項目
echo "可以安全刪除的項目:"
echo ""

# 不完整的解壓縮
if [ -d "$EXTRACT_DIR" ]; then
    EXTRACT_COUNT=$(ls -d "$EXTRACT_DIR"/*/ 2>/dev/null | wc -l)
    if [ "$EXTRACT_COUNT" -lt 10 ]; then
        echo "1. ❌ 不完整的解壓縮目錄 (~28MB, 只有 2 個 cells):"
        echo "   rm -rf $EXTRACT_DIR"
        echo ""
    fi
fi

# __pycache__
if [ -d "$IVF_ANALYSIS_DIR/__pycache__" ]; then
    echo "2. 🗑️  Python cache (__pycache__):"
    echo "   rm -rf $IVF_ANALYSIS_DIR/__pycache__"
    echo ""
fi

# 關於 12GB tar.gz 檔案
echo "============================================================"
echo "6. 關於 12GB tar.gz 檔案的使用"
echo "============================================================"
echo ""

if [ -f "$TAR_FILE" ]; then
    echo "✓ 你已經有一個 12GB 的 tar.gz 檔案"
    echo ""
    echo "選項 1: 直接使用 tar.gz (不需要解壓)"
    echo "   - 可以從 tar.gz 直接讀取特定檔案"
    echo "   - 但需要每次都要解壓縮，比較慢"
    echo ""
    echo "選項 2: 解壓到 staging (推薦)"
    echo "   - 如果 staging 有足夠空間，解壓後使用會更快"
    echo "   - 但需要確保有足夠的 quota"
    echo ""
    echo "選項 3: 只解壓需要的 cells"
    echo "   - 可以只解壓特定幾個 cells (例如: RI382-2, ZS435-5, RS363-7)"
    echo "   - 這樣可以節省空間"
    echo ""
    
    echo "要只解壓特定 cells，可以用:"
    echo "cd $IVF_DATA_DIR"
    echo "tar -xzvf embryo_dataset.tar.gz embryo_dataset/RI382-2 embryo_dataset/ZS435-5 embryo_dataset/RS363-7"
    echo ""
fi

echo "============================================================"
echo "完成！"
echo "============================================================"

