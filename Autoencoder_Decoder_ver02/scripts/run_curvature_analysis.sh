#!/bin/bash
# 運行 curvature analysis 的指令（使用 group GPU）

# 切換到正確的目錄
cd /staging/groups/bhaskar_group/rho9/ivf_analysis

# 檢查 GPU 是否可用
echo "============================================================"
echo "Checking GPU availability..."
echo "============================================================"
nvidia-smi || echo "⚠️  nvidia-smi not available (may still work)"
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "⚠️  PyTorch not available or CUDA not detected"

# ============================================================
# 基本命令（使用默認值：PCA + GPU）
# ============================================================

echo ""
echo "============================================================"
echo "Running analysis for ZS435-5 (good quality)"
echo "============================================================"

python3 scripts/analyze_trajectory_curvature.py \
    --video_name ZS435-5 \
    --method pca \
    --device cuda

echo ""
echo "============================================================"
echo "Running analysis for RS363-7 (poor quality)"
echo "============================================================"

python3 scripts/analyze_trajectory_curvature.py \
    --video_name RS363-7 \
    --method pca \
    --device cuda

# ============================================================
# 如果沒有 GPU，使用 CPU
# ============================================================

# python3 scripts/analyze_trajectory_curvature.py \
#     --video_name ZS435-5 \
#     --method pca \
#     --device cpu

# ============================================================
# 完整參數示例（如果需要自定義）
# ============================================================

# python3 scripts/analyze_trajectory_curvature.py \
#     --video_name ZS435-5 \
#     --method pca \
#     --device cuda \
#     --checkpoint /home/rho9/ivf_repo/checkpoints/checkpoint_epoch_50.pt \
#     --data_root /staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz \
#     --output_dir /staging/groups/bhaskar_group/rho9/curvature_analysis \
#     --curvature_threshold_percentile 95.0 \
#     --max_frames None

