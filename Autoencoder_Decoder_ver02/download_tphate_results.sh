#!/bin/bash
# 下载 TPHATE 结果到本地 Mac

echo "=== Downloading TPHATE Results ==="
echo ""

# 设置变量
REMOTE_USER="rho9"
REMOTE_HOST="ap2001.chtc.wisc.edu"
REMOTE_DIR="~/ivf_repo"
LOCAL_DIR="$HOME/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02/tphate_results"

# 创建本地目录
mkdir -p "$LOCAL_DIR"

echo "Downloading from: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
echo "Saving to: $LOCAL_DIR"
echo ""

# 下载 TPHATE 结果文件
echo "1. Downloading TPHATE results..."
scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/tphate_3d_results_direct.npz "$LOCAL_DIR/" 2>/dev/null && echo "   ✓ tphate_3d_results_direct.npz" || echo "   ⚠️  tphate_3d_results_direct.npz not found"

# 下载 latent 文件
echo ""
echo "2. Downloading latent files..."
scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/latents_all_frames_direct.npz "$LOCAL_DIR/" 2>/dev/null && echo "   ✓ latents_all_frames_direct.npz" || echo "   ⚠️  latents_all_frames_direct.npz not found"
scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/latents_preprocessed_direct.npz "$LOCAL_DIR/" 2>/dev/null && echo "   ✓ latents_preprocessed_direct.npz" || echo "   ⚠️  latents_preprocessed_direct.npz not found"

# 下载可视化结果
echo ""
echo "3. Downloading visualizations..."
# 尝试多种可能的路径
if scp -r ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/tphate_segments_direct "$LOCAL_DIR/" 2>/dev/null; then
    echo "   ✓ tphate_segments_direct/ (all visualizations)"
elif scp -r ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/tphate_segments_direct/* "$LOCAL_DIR/tphate_segments_direct/" 2>/dev/null; then
    echo "   ✓ tphate_segments_direct/* (all visualizations)"
else
    echo "   ⚠️  tphate_segments_direct/ not found"
    echo "   Trying to list remote directory..."
    ssh ${REMOTE_USER}@${REMOTE_HOST} "ls -lh ${REMOTE_DIR}/tphate_segments_direct/" 2>/dev/null || echo "   Directory does not exist on remote"
fi

# 下载元数据
echo ""
echo "4. Downloading metadata..."
scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/latents_all_frames_direct.json "$LOCAL_DIR/" 2>/dev/null && echo "   ✓ latents_all_frames_direct.json" || echo "   ⚠️  latents_all_frames_direct.json not found"

echo ""
echo "=========================================="
echo "✅ Download Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $LOCAL_DIR"
echo ""
echo "Files downloaded:"
ls -lh "$LOCAL_DIR" 2>/dev/null | tail -n +2
echo ""
echo "Visualizations:"
ls -lh "$LOCAL_DIR/tphate_segments_direct" 2>/dev/null | tail -n +2 || echo "  (check if directory exists)"

