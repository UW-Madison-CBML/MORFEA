#!/bin/bash
# 簡單的下載腳本

cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

REMOTE="rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/curvature_analysis"

echo "下載 Good Quality (ZS435-5)..."
scp "$REMOTE/figures/tphate_curvature_ZS435-5.png" curvature_results/good/figures/ 2>/dev/null && echo "  ✓ tphate_curvature_ZS435-5.png" || echo "  ⚠️  跳過"
scp "$REMOTE/figures/high_curvature_montage_ZS435-5.png" curvature_results/good/figures/ 2>/dev/null && echo "  ✓ montage" || echo "  ⚠️  跳過"
scp "$REMOTE/curvature_data_ZS435-5.npz" curvature_results/good/ 2>/dev/null && echo "  ✓ data" || echo "  ⚠️  跳過"
scp "$REMOTE/frames/high_curvature/ZS435-5"* curvature_results/good/frames/high_curvature/ 2>/dev/null && echo "  ✓ frames" || echo "  ⚠️  跳過"

echo ""
echo "下載 Poor Quality (RS363-7)..."
scp "$REMOTE/figures/tphate_curvature_RS363-7.png" curvature_results/bad/figures/ 2>/dev/null && echo "  ✓ tphate_curvature_RS363-7.png" || echo "  ⚠️  跳過"
scp "$REMOTE/figures/high_curvature_montage_RS363-7.png" curvature_results/bad/figures/ 2>/dev/null && echo "  ✓ montage" || echo "  ⚠️  跳過"
scp "$REMOTE/curvature_data_RS363-7.npz" curvature_results/bad/ 2>/dev/null && echo "  ✓ data" || echo "  ⚠️  跳過"
scp "$REMOTE/frames/high_curvature/RS363-7"* curvature_results/bad/frames/high_curvature/ 2>/dev/null && echo "  ✓ frames" || echo "  ⚠️  跳過"

echo ""
echo "✓ 下載完成！"
echo "結果在: curvature_results/good/ 和 curvature_results/bad/"

