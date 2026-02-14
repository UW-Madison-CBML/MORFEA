# 完整運行步驟

## 步驟 1: 上傳文件到 CHTC（在本地運行）

```bash
cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

# 上傳所有需要的文件
scp scripts/analyze_trajectory_curvature.py \
    rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/ivf_analysis/scripts/

scp scripts/run_curvature_analysis.sh \
    rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/ivf_analysis/scripts/

scp scripts/run_curvature_single.sh \
    rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/ivf_analysis/scripts/

scp scripts/curvature_analysis_gpu.submit \
    rho9@ap2001.chtc.wisc.edu:/home/rho9/
```

## 步驟 2: 在 CHTC 上運行（兩種方法）

### 方法 A: 使用 HTCondor 提交作業（推薦用於批量）

```bash
# SSH 到 CHTC
ssh rho9@ap2001.chtc.wisc.edu

# 切換到 home 目錄
cd ~

# 提交作業
condor_submit curvature_analysis_gpu.submit

# 監控作業狀態
condor_q

# 查看日誌
tail -f ~/curvature_analysis_ZS435-5.log
```

### 方法 B: 直接運行（更簡單，推薦）

```bash
# SSH 到 CHTC
ssh rho9@ap2001.chtc.wisc.edu

# 切換到分析目錄
cd /staging/groups/bhaskar_group/rho9/ivf_analysis

# 直接運行（會自動使用 GPU）
bash scripts/run_curvature_analysis.sh
```

## 步驟 3: 檢查結果

```bash
# 查看輸出目錄
ls -lh /staging/groups/bhaskar_group/rho9/curvature_analysis/figures/

# 應該看到：
# - pca_curvature_ZS435-5.png
# - pca_curvature_RS363-7.png
# - high_curvature_montage_ZS435-5.png
# - high_curvature_montage_RS363-7.png
# - frames/high_curvature/ 目錄（包含高 curvature frames）
```

## 步驟 4: 下載結果到本地（可選）

```bash
# 在本地運行
cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

# 創建下載目錄
mkdir -p results/curvature_analysis

# 下載結果
scp -r rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/curvature_analysis/* \
    results/curvature_analysis/
```

## 驗證 GPU 使用

運行時應該看到：

```
✓ GPU available: NVIDIA GeForce RTX 3090
  GPU memory: 24.00 GB
Method: PCA
Device: cuda
  ✓ Using GPU acceleration
```

## 如果遇到問題

### GPU 不可用
- 檢查：`nvidia-smi`
- 腳本會自動回退到 CPU

### 文件找不到
- 確認文件已上傳：`ls -lh /staging/groups/bhaskar_group/rho9/ivf_analysis/scripts/`

### HTCondor 提交失敗
- 確保從 home 目錄提交：`cd ~`
- 或使用直接運行方法（方法 B）

