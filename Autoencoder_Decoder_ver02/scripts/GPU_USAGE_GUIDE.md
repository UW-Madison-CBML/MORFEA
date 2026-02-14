# 使用 Group GPU 的完整指南

## 方法 1: 直接運行（如果可以直接訪問 GPU 節點）

```bash
# 在 CHTC 上直接運行
cd /staging/groups/bhaskar_group/rho9/ivf_analysis

# 檢查 GPU 是否可用
nvidia-smi

# 運行分析（會自動使用 GPU）
bash scripts/run_curvature_analysis.sh
```

## 方法 2: 使用 HTCondor 提交作業（推薦）

### 步驟 1: 上傳文件到 CHTC

```bash
# 從本地上傳
scp scripts/analyze_trajectory_curvature.py \
    rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/ivf_analysis/scripts/

scp scripts/run_curvature_single.sh \
    rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/ivf_analysis/scripts/

scp scripts/curvature_analysis_gpu.submit \
    rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/ivf_analysis/
```

### 步驟 2: 在 CHTC 上提交作業

```bash
# SSH 到 CHTC
ssh rho9@ap2001.chtc.wisc.edu

# 切換到目錄
cd /staging/groups/bhaskar_group/rho9/ivf_analysis

# 提交作業
condor_submit scripts/curvature_analysis_gpu.submit
```

### 步驟 3: 監控作業

```bash
# 查看作業狀態
condor_q

# 查看詳細信息
condor_q -better-analyze

# 查看日誌
tail -f curvature_analysis_ZS435-5.log
```

## 方法 3: 交互式會話請求 GPU

```bash
# 請求交互式 GPU 節點
condor_submit -interactive -request-gpus 1

# 或者使用
condor_submit -interactive \
    -append "request_gpus = 1" \
    -append "request_cpus = 4" \
    -append "request_memory = 8GB"
```

## 驗證 GPU 使用

運行時應該看到：

```
============================================================
Checking GPU availability...
============================================================
✓ GPU available: NVIDIA GeForce RTX 3090
  GPU memory: 24.00 GB

============================================================
Trajectory Curvature Analysis
============================================================
Video: ZS435-5
Method: PCA
Device: cuda
  ✓ Using GPU acceleration
```

## 常見問題

### Q: 如何確認 GPU 是否可用？

```bash
# 方法 1: 使用 nvidia-smi
nvidia-smi

# 方法 2: 使用 Python
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Q: 如果沒有 GPU 怎麼辦？

腳本會自動回退到 CPU，但會顯示警告：
```
⚠️  WARNING: CUDA requested but not available. Falling back to CPU.
  ⚠ Using CPU (slower)
```

### Q: 如何檢查作業是否真的在使用 GPU？

查看日誌文件中的輸出：
```bash
grep "GPU" curvature_analysis_*.log
grep "cuda" curvature_analysis_*.log
```

## 性能對比

- **CPU**: ~10-30 分鐘 per embryo
- **GPU**: ~1-3 分鐘 per embryo

使用 GPU 可以快 **10-30 倍**！

