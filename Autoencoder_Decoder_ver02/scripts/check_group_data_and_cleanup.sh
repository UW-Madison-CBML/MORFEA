#!/bin/bash
# 檢查 group 層級是否有資料，然後決定是否清理用戶的 ivf_data

set -e

echo "============================================================"
echo "檢查 Group 資料和清理建議"
echo "============================================================"
echo ""

# 檢查 group 層級的 ivf 資料
GROUP_IVF_DIR="/staging/groups/bhaskar_group/ivf"
USER_IVF_DATA="/staging/groups/bhaskar_group/rho9/ivf_data"

echo "1. 檢查 Group 層級的 IVF 資料:"
echo "============================================================"
if [ -d "$GROUP_IVF_DIR" ]; then
    echo "✓ 找到 group 層級的 IVF 目錄: $GROUP_IVF_DIR"
    echo ""
    
    echo "內容:"
    ls -lh "$GROUP_IVF_DIR" 2>/dev/null | head -20
    echo ""
    
    # 檢查大小
    GROUP_SIZE=$(du -sh "$GROUP_IVF_DIR" 2>/dev/null | cut -f1)
    echo "總大小: $GROUP_SIZE"
    echo ""
    
    # 檢查是否有 embryo_dataset
    if [ -f "$GROUP_IVF_DIR/embryo_dataset.tar.gz" ] || [ -d "$GROUP_IVF_DIR/embryo_dataset" ]; then
        echo "✓ Group 層級有 embryo_dataset"
        if [ -f "$GROUP_IVF_DIR/embryo_dataset.tar.gz" ]; then
            TAR_SIZE=$(du -sh "$GROUP_IVF_DIR/embryo_dataset.tar.gz" | cut -f1)
            echo "  tar.gz 大小: $TAR_SIZE"
        fi
        if [ -d "$GROUP_IVF_DIR/embryo_dataset" ]; then
            EXTRACT_SIZE=$(du -sh "$GROUP_IVF_DIR/embryo_dataset" 2>/dev/null | cut -f1)
            EXTRACT_COUNT=$(ls -d "$GROUP_IVF_DIR/embryo_dataset"/*/ 2>/dev/null | wc -l)
            echo "  已解壓目錄大小: $EXTRACT_SIZE"
            echo "  Cell 目錄數量: $EXTRACT_COUNT"
        fi
        echo ""
    else
        echo "⚠️  Group 層級沒有 embryo_dataset"
        echo ""
    fi
else
    echo "⚠️  Group 層級的 IVF 目錄不存在: $GROUP_IVF_DIR"
    echo ""
fi

echo "2. 檢查用戶的 IVF 資料:"
echo "============================================================"
if [ -d "$USER_IVF_DATA" ]; then
    echo "位置: $USER_IVF_DATA"
    USER_SIZE=$(du -sh "$USER_IVF_DATA" 2>/dev/null | cut -f1)
    echo "總大小: $USER_SIZE"
    echo ""
    
    echo "內容:"
    ls -lh "$USER_IVF_DATA" 2>/dev/null
    echo ""
    
    # 檢查 tar.gz
    if [ -f "$USER_IVF_DATA/embryo_dataset.tar.gz" ]; then
        TAR_SIZE=$(du -sh "$USER_IVF_DATA/embryo_dataset.tar.gz" | cut -f1)
        echo "  tar.gz: $TAR_SIZE"
    fi
    
    # 檢查已解壓目錄
    if [ -d "$USER_IVF_DATA/embryo_dataset" ]; then
        EXTRACT_SIZE=$(du -sh "$USER_IVF_DATA/embryo_dataset" 2>/dev/null | cut -f1)
        EXTRACT_COUNT=$(ls -d "$USER_IVF_DATA/embryo_dataset"/*/ 2>/dev/null | wc -l)
        echo "  已解壓目錄: $EXTRACT_SIZE ($EXTRACT_COUNT cells)"
    fi
    echo ""
else
    echo "⚠️  用戶的 IVF 資料目錄不存在"
    echo ""
fi

echo "3. 清理建議:"
echo "============================================================"
echo ""

# 如果 group 有完整的資料，可以刪除用戶的
if [ -d "$GROUP_IVF_DIR" ] && ([ -f "$GROUP_IVF_DIR/embryo_dataset.tar.gz" ] || [ -d "$GROUP_IVF_DIR/embryo_dataset" ]); then
    echo "✓ Group 層級有 IVF 資料"
    echo ""
    
    if [ -d "$USER_IVF_DATA" ]; then
        echo "可以刪除用戶的 ivf_data 來釋放空間:"
        echo ""
        
        # 檢查可以刪除什麼
        if [ -f "$USER_IVF_DATA/embryo_dataset.tar.gz" ]; then
            TAR_SIZE=$(du -sh "$USER_IVF_DATA/embryo_dataset.tar.gz" | cut -f1)
            echo "1. 刪除 tar.gz (${TAR_SIZE}):"
            echo "   rm -f $USER_IVF_DATA/embryo_dataset.tar.gz"
            echo ""
        fi
        
        if [ -d "$USER_IVF_DATA/embryo_dataset" ]; then
            EXTRACT_SIZE=$(du -sh "$USER_IVF_DATA/embryo_dataset" 2>/dev/null | cut -f1)
            echo "2. 刪除已解壓目錄 (${EXTRACT_SIZE}):"
            echo "   rm -rf $USER_IVF_DATA/embryo_dataset"
            echo ""
        fi
        
        echo "3. 或者刪除整個 ivf_data 目錄:"
        echo "   rm -rf $USER_IVF_DATA"
        echo ""
        
        read -p "要刪除用戶的 ivf_data 嗎? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "刪除中..."
            rm -rf "$USER_IVF_DATA"
            echo "✓ 已刪除: $USER_IVF_DATA"
            echo ""
            
            # 檢查釋放的空間
            echo "檢查釋放的空間:"
            du -sh /staging/groups/bhaskar_group/rho9/
        else
            echo "保留用戶的 ivf_data"
        fi
    else
        echo "用戶的 ivf_data 目錄不存在，無需刪除"
    fi
else
    echo "⚠️  Group 層級沒有完整的 IVF 資料"
    echo "   建議保留用戶的 ivf_data"
fi

echo ""
echo "============================================================"
echo "完成！"
echo "============================================================"

