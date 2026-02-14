#!/bin/bash
# 专门下载 TPHATE 可视化结果

REMOTE_USER="rho9"
REMOTE_HOST="ap2001.chtc.wisc.edu"
REMOTE_DIR="~/ivf_repo"
LOCAL_DIR="$HOME/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02/tphate_results"

mkdir -p "$LOCAL_DIR/tphate_segments_direct"

echo "=== Downloading TPHATE Visualizations ==="
echo ""

# 列出远程文件
echo "Checking remote files..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "ls -lh ${REMOTE_DIR}/tphate_segments_direct/" 2>/dev/null

echo ""
echo "Downloading files..."

# 下载所有 PNG 文件
scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/tphate_segments_direct/*.png "$LOCAL_DIR/tphate_segments_direct/" 2>/dev/null && echo "✓ PNG files downloaded" || echo "⚠️  PNG files not found"

# 下载 JSON 文件
scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/tphate_segments_direct/*.json "$LOCAL_DIR/tphate_segments_direct/" 2>/dev/null && echo "✓ JSON files downloaded" || echo "⚠️  JSON files not found"

echo ""
echo "Local files:"
ls -lh "$LOCAL_DIR/tphate_segments_direct/" 2>/dev/null || echo "  (directory is empty)"
