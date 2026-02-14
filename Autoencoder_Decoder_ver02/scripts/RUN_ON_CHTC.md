# 在 CHTC 上運行的完整步驟

## 步驟 1: SSH 到 CHTC

在本地終端運行：

```bash
ssh rho9@ap2001.chtc.wisc.edu
```

輸入密碼和 Duo 驗證。

## 步驟 2: 切換到分析目錄

```bash
cd /staging/groups/bhaskar_group/rho9/ivf_analysis
```

## 步驟 3: 運行分析

### 方法 A: 使用腳本（推薦）

```bash
bash scripts/run_curvature_analysis.sh
```

這會自動分析兩個 embryo（ZS435-5 和 RS363-7）。

### 方法 B: 單個運行

```bash
# 分析 ZS435-5
python3 scripts/analyze_trajectory_curvature.py \
    --video_name ZS435-5 \
    --method pca \
    --device cuda

# 分析 RS363-7
python3 scripts/analyze_trajectory_curvature.py \
    --video_name RS363-7 \
    --method pca \
    --device cuda
```

## 步驟 4: 檢查結果

```bash
# 查看生成的圖表
ls -lh /staging/groups/bhaskar_group/rho9/curvature_analysis/figures/

# 應該看到：
# - pca_curvature_ZS435-5.png
# - pca_curvature_RS363-7.png
# - high_curvature_montage_ZS435-5.png
# - high_curvature_montage_RS363-7.png
```

## 重要提示

- `/staging/` 路徑只在 CHTC 服務器上存在
- 必須先 SSH 到 CHTC 才能運行這些命令
- 腳本會自動檢測 GPU，如果沒有則使用 CPU

