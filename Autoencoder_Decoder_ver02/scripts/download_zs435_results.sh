#!/bin/bash
# 下載 ZS435-5 的 PCA 分析結果到本地

CHTC_USER="rho9"
CHTC_HOST="ap2001.chtc.wisc.edu"
CHTC_DIR="/staging/groups/bhaskar_group/rho9/curvature_analysis"
LOCAL_DIR="/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02/results/curvature_analysis/ZS435-5"

echo "============================================================"
echo "Downloading ZS435-5 PCA results from CHTC..."
echo "============================================================"

# 創建本地目錄
mkdir -p "$LOCAL_DIR/figures"
mkdir -p "$LOCAL_DIR/frames/high_curvature"

# 下載所有 ZS435-5 相關的圖表
echo "Downloading all ZS435-5 figures..."
scp ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/figures/*ZS435-5*.png \
    "$LOCAL_DIR/figures/" 2>&1 | grep -v "not found" && echo "✓ Downloaded figures" || echo "⚠️  Some figures may not be found"

# 下載 PCA 軌跡圖（明確指定）
echo "Downloading pca_curvature_ZS435-5.png..."
scp ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/figures/pca_curvature_ZS435-5.png \
    "$LOCAL_DIR/figures/" 2>&1 | grep -v "not found" && echo "✓ Downloaded pca_curvature_ZS435-5.png" || echo "⚠️  pca_curvature_ZS435-5.png not found (may still be running)"

# 下載 high curvature montage
echo "Downloading high_curvature_montage_ZS435-5.png..."
scp ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/figures/high_curvature_montage_ZS435-5.png \
    "$LOCAL_DIR/figures/" 2>&1 | grep -v "not found" && echo "✓ Downloaded high_curvature_montage_ZS435-5.png" || echo "⚠️  high_curvature_montage_ZS435-5.png not found"

# 下載所有高 curvature frames
echo "Downloading high curvature frames..."
scp ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/frames/high_curvature/ZS435-5_*.png \
    "$LOCAL_DIR/frames/high_curvature/" 2>&1 | grep -v "not found" && echo "✓ Downloaded high curvature frames" || echo "⚠️  High curvature frames not found"

# 下載 curvature 數據（如果有的話）
echo "Downloading curvature data..."
scp ${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/curvature_data_ZS435-5.json \
    "$LOCAL_DIR/" 2>/dev/null && echo "✓ Downloaded curvature data" || echo "⚠️  Curvature data not found"

echo ""
echo "============================================================"
echo "Download complete!"
echo "============================================================"
echo "Files saved to: $LOCAL_DIR"
echo ""
echo "Downloaded files:"
ls -lh "$LOCAL_DIR/figures/" 2>/dev/null
echo ""
echo "High curvature frames:"
ls -lh "$LOCAL_DIR/frames/high_curvature/" 2>/dev/null | head -10
echo "..."

