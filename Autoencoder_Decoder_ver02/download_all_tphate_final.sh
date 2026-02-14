#!/bin/bash
# 下载所有 TPHATE 可视化文件

REMOTE_USER="rho9"
REMOTE_HOST="ap2001.chtc.wisc.edu"
REMOTE_DIR="~/ivf_repo/tphate_segments_direct"
LOCAL_DIR="$HOME/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02/tphate_results/tphate_segments_direct"

mkdir -p "$LOCAL_DIR"

echo "=== Downloading All TPHATE Visualizations ==="
echo ""

files=(
    "tphate_3d_gradient.png"
    "tphate_3d_segments.png"
    "all_segments_all_frames.png"
    "segment_A_frames.png"
    "segment_B_frames.png"
    "segment_C_frames.png"
    "segment_D_frames.png"
    "tphate_segments_combined.png"
    "segments_metadata.json"
)

success_count=0
fail_count=0

for file in "${files[@]}"; do
    echo -n "Downloading $file... "
    if scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${file} "$LOCAL_DIR/" 2>/dev/null; then
        size=$(stat -f%z "$LOCAL_DIR/$file" 2>/dev/null || stat -c%s "$LOCAL_DIR/$file" 2>/dev/null)
        size_mb=$(echo "scale=2; $size/1024/1024" | bc 2>/dev/null || echo "?")
        echo "✓ (${size_mb} MB)"
        success_count=$((success_count + 1))
    else
        echo "✗"
        fail_count=$((fail_count + 1))
    fi
done

echo ""
echo "=========================================="
echo "Download Summary:"
echo "  ✓ Success: $success_count"
echo "  ✗ Failed: $fail_count"
echo "=========================================="
echo ""
echo "Downloaded files:"
ls -lh "$LOCAL_DIR" 2>/dev/null | tail -n +2

