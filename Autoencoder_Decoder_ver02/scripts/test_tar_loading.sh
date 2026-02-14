#!/bin/bash
# 測試從 tar.gz 直接讀取

echo "============================================================"
echo "測試從 tar.gz 直接讀取"
echo "============================================================"
echo ""

cd /staging/groups/bhaskar_group/rho9/ivf_analysis

# 檢查腳本是否存在
if [ ! -f "scripts/analyze_trajectory_curvature.py" ]; then
    echo "❌ 腳本不存在，請先上傳"
    exit 1
fi

# 檢查 tar.gz 是否存在
TAR_FILE="/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz"
if [ ! -f "$TAR_FILE" ]; then
    echo "❌ tar.gz 不存在: $TAR_FILE"
    exit 1
fi

echo "✓ tar.gz 存在: $TAR_FILE"
echo ""

# 測試檢查特定 cell 是否存在
echo "檢查 ZS435-5 是否存在於 tar.gz..."
tar -tzf "$TAR_FILE" | grep "^embryo_dataset/ZS435-5/" | head -3
if [ $? -eq 0 ]; then
    echo "✓ ZS435-5 存在"
else
    echo "⚠️  ZS435-5 不存在"
fi

echo ""
echo "檢查 RS363-7 是否存在於 tar.gz..."
tar -tzf "$TAR_FILE" | grep "^embryo_dataset/RS363-7/" | head -3
if [ $? -eq 0 ]; then
    echo "✓ RS363-7 存在"
else
    echo "⚠️  RS363-7 不存在"
fi

echo ""
echo "============================================================"
echo "現在可以執行分析（會自動從 tar.gz 讀取）"
echo "============================================================"
echo ""
echo "python3 scripts/analyze_trajectory_curvature.py --video_name ZS435-5"
echo "python3 scripts/analyze_trajectory_curvature.py --video_name RS363-7"
echo ""

