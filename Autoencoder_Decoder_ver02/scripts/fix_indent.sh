#!/bin/bash
# 修復縮排錯誤

cd /staging/groups/bhaskar_group/rho9/ivf_analysis/scripts

# 使用 sed 修復縮排
# 找到 group_tar 那行，將它改為正確的縮排（8個空格）
sed -i 's/^[[:space:]]*group_tar = Path/        group_tar = Path/' analyze_trajectory_curvature.py

echo "✓ 已修復縮排"
echo ""
echo "驗證修復結果:"
sed -n '595,605p' analyze_trajectory_curvature.py

