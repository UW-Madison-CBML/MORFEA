#!/bin/bash
# 重试下载 TPHATE 可视化文件（处理网络超时）

REMOTE_USER="rho9"
REMOTE_HOST="ap2001.chtc.wisc.edu"
REMOTE_DIR="~/ivf_repo/tphate_segments_direct"
LOCAL_DIR="$HOME/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02/tphate_results/tphate_segments_direct"

mkdir -p "$LOCAL_DIR"

echo "=== Retrying Download of TPHATE Visualizations ==="
echo ""

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

max_retries=3
retry_delay=5

for file in "${files[@]}"; do
    # 检查文件是否已存在
    if [ -f "$LOCAL_DIR/$file" ]; then
        size=$(stat -f%z "$LOCAL_DIR/$file" 2>/dev/null || stat -c%s "$LOCAL_DIR/$file" 2>/dev/null)
        if [ "$size" -gt 1000 ]; then
            echo "✓ $file (already downloaded, ${size} bytes)"
            continue
        fi
    fi
    
    echo -n "Downloading $file... "
    retry=0
    success=false
    
    while [ $retry -lt $max_retries ]; do
        if scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${file} "$LOCAL_DIR/" 2>/dev/null; then
            echo "✓"
            success=true
            break
        else
            retry=$((retry + 1))
            if [ $retry -lt $max_retries ]; then
                echo -n "(retry $retry/$max_retries in ${retry_delay}s...) "
                sleep $retry_delay
            fi
        fi
    done
    
    if [ "$success" = false ]; then
        echo "✗ (failed after $max_retries retries)"
    fi
done

echo ""
echo "=========================================="
echo "Download Summary:"
echo "=========================================="
ls -lh "$LOCAL_DIR" 2>/dev/null | tail -n +2

