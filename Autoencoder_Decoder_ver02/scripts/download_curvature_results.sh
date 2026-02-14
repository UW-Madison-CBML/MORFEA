#!/bin/bash
# 下載 curvature 分析結果並組織到 good/bad 文件夾

set -e

echo "============================================================"
echo "下載 Curvature 分析結果"
echo "============================================================"
echo ""

# 本地目錄
LOCAL_BASE="/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02/curvature_results"
GOOD_DIR="$LOCAL_BASE/good"
BAD_DIR="$LOCAL_BASE/bad"

# 遠端目錄
REMOTE_BASE="rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/curvature_analysis"

# 創建本地目錄
mkdir -p "$GOOD_DIR/figures"
mkdir -p "$GOOD_DIR/frames/high_curvature"
mkdir -p "$BAD_DIR/figures"
mkdir -p "$BAD_DIR/frames/high_curvature"

echo "本地目錄結構："
echo "  Good (ZS435-5): $GOOD_DIR"
echo "  Bad (RS363-7): $BAD_DIR"
echo ""

# Good quality embryo: ZS435-5
echo "============================================================"
echo "下載 Good Quality Embryo (ZS435-5)"
echo "============================================================"
echo ""

GOOD_FILES=(
    "figures/tphate_curvature_ZS435-5.png"
    "figures/high_curvature_montage_ZS435-5.png"
    "curvature_data_ZS435-5.npz"
)

for file in "${GOOD_FILES[@]}"; do
    echo "下載: $file"
    scp "$REMOTE_BASE/$file" "$GOOD_DIR/$file" 2>/dev/null && echo "  ✓ 完成" || echo "  ⚠️  跳過（可能不存在）"
done

# 下載 high-curvature frames
echo ""
echo "下載 high-curvature frames..."
scp -r "$REMOTE_BASE/frames/high_curvature/ZS435-5"* "$GOOD_DIR/frames/high_curvature/" 2>/dev/null && echo "  ✓ 完成" || echo "  ⚠️  跳過"

# Bad quality embryo: RS363-7
echo ""
echo "============================================================"
echo "下載 Poor Quality Embryo (RS363-7)"
echo "============================================================"
echo ""

BAD_FILES=(
    "figures/tphate_curvature_RS363-7.png"
    "figures/high_curvature_montage_RS363-7.png"
    "curvature_data_RS363-7.npz"
)

for file in "${BAD_FILES[@]}"; do
    echo "下載: $file"
    scp "$REMOTE_BASE/$file" "$BAD_DIR/$file" 2>/dev/null && echo "  ✓ 完成" || echo "  ⚠️  跳過（可能不存在）"
done

# 下載 high-curvature frames
echo ""
echo "下載 high-curvature frames..."
scp -r "$REMOTE_BASE/frames/high_curvature/RS363-7"* "$BAD_DIR/frames/high_curvature/" 2>/dev/null && echo "  ✓ 完成" || echo "  ⚠️  跳過"

echo ""
echo "============================================================"
echo "下載完成！"
echo "============================================================"
echo ""
echo "結果位置："
echo "  Good (ZS435-5): $GOOD_DIR"
echo "  Bad (RS363-7): $BAD_DIR"
echo ""
echo "檢查下載的文件："
echo "  ls -lh $GOOD_DIR/figures/"
echo "  ls -lh $BAD_DIR/figures/"
echo ""

