#!/bin/bash
# 下载所有 TPHATE 结果（包括可视化）

REMOTE_USER="rho9"
REMOTE_HOST="ap2001.chtc.wisc.edu"
REMOTE_DIR="~/ivf_repo"
LOCAL_DIR="$HOME/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02/tphate_results"

mkdir -p "$LOCAL_DIR/tphate_segments_direct"

echo "=== Downloading All TPHATE Results ==="
echo ""

# 1. 检查远程目录
echo "1. Checking remote directory structure..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR} && ls -lh tphate_segments_direct/ 2>/dev/null || echo 'Directory not found'"

echo ""
echo "2. Downloading visualization files..."

# 下载所有可视化文件（逐个尝试）
files=(
    "tphate_3d_gradient.png"
    "tphate_3d_segments.png"
    "segment_A_frames.png"
    "segment_B_frames.png"
    "segment_C_frames.png"
    "segment_D_frames.png"
    "tphate_segments_combined.png"
    "segments_metadata.json"
)

for file in "${files[@]}"; do
    echo -n "  Downloading $file... "
    if scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/tphate_segments_direct/${file} "$LOCAL_DIR/tphate_segments_direct/" 2>/dev/null; then
        echo "✓"
    else
        echo "✗ (not found)"
    fi
done

echo ""
echo "3. Checking downloaded files..."
ls -lh "$LOCAL_DIR/tphate_segments_direct/" 2>/dev/null

echo ""
echo "=========================================="
echo "✅ Download Complete!"
echo "=========================================="

