#!/bin/bash
# 單個 embryo 的 curvature analysis 腳本（用於 HTCondor）

VIDEO_NAME=$1

# 切換到正確的目錄
cd /staging/groups/bhaskar_group/rho9/ivf_analysis

# 檢查 GPU 是否可用
echo "============================================================"
echo "Checking GPU availability..."
echo "============================================================"
nvidia-smi || echo "⚠️  nvidia-smi not available"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 運行分析
echo "============================================================"
echo "Running curvature analysis for: $VIDEO_NAME"
echo "============================================================"

# 自動檢測 GPU，如果沒有則使用 CPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo "✓ GPU detected, using CUDA"
else
    DEVICE="cpu"
    echo "⚠️  No GPU detected, using CPU"
fi

python3 scripts/analyze_trajectory_curvature.py \
    --video_name "$VIDEO_NAME" \
    --method pca \
    --device "$DEVICE"

echo "============================================================"
echo "Analysis completed for: $VIDEO_NAME"
echo "============================================================"

