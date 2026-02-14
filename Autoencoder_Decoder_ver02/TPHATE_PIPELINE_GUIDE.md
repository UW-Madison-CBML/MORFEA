# 嚴謹版 3D T-PHATE Pipeline 指南

## 概述

這個 pipeline 實現了嚴謹的 3D T-PHATE 分析流程，從 latent vector 提取到最終的 3D 可視化和分析。

## Pipeline 步驟

### Step 0: 決定要餵給 TPHATE 的點

**每個 cell 的每個 frame** 都是一個點：
- Encoder 輸入：單張 frame (128×128)
- Latent 維度：D = 256 (根據 model.py 的 `encoder_hidden_dim`)
- 每個 clip：16 個 frame（但我們提取每個 frame 的 latent）

**TPHATE 看到的點雲：**
- N_frames = (#cell × 每 cell 的 frame 數)
- 每個點是 z ∈ R^D，外加 metadata：哪個 cell？第幾幀？絕對時間？

### Step 1: 用 epoch 50 的 encoder 導出所有 latent

**腳本：** `export_all_frame_latents.py`

**功能：**
- 載入 `checkpoint_epoch_50.pt`，只使用 encoder 部分
- 對 `index.csv` 裡所有序列的**每一幀**提取 latent
- 輸出：`latents_all_frames.npz`

**使用方法：**
```bash
python3 export_all_frame_latents.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --index_csv index.csv \
    --output latents_all_frames.npz \
    --max_sequences 100  # 測試時限制，正式運行改為 None
```

**輸出格式：**
```python
{
    'Z': [N_frames, D],           # Latent vectors
    'cell_id': [N_frames],        # Cell IDs
    'frame_in_cell': [N_frames],  # Frame index within cell
    'abs_time': [N_frames],       # Absolute time (RUN number)
    'sequence_idx': [N_frames],   # Original sequence index
    'paths': [N_frames]           # Image paths
}
```

### Step 2: 對 latent 做乾淨的 pre-processing

**腳本：** `preprocess_latents.py`

**功能：**
1. **移除異常值**：去掉明顯問題樣本（可選）
2. **標準化**：z-score normalization
3. **PCA 降維**（可選）：降到 20-50 維去噪

**使用方法：**
```bash
python3 preprocess_latents.py \
    --input latents_all_frames.npz \
    --output latents_preprocessed.npz \
    --pca_components 32 \
    --outlier_threshold 5.0
```

**輸出格式：**
```python
{
    'Z_norm': [N_frames, D],      # 標準化後的 latent
    'Z_pca': [N_frames, 32],      # PCA 降維後的 latent（如果啟用）
    'cell_id': [N_frames],
    'frame_in_cell': [N_frames],
    'abs_time': [N_frames],
    'sequence_idx': [N_frames]
}
```

### Step 3: 把時間資訊 encode 給 TPHATE

**腳本：** `tphate_3d_pipeline.py` (包含 Step 3-4)

**功能：**
- 建立時間鄰接關係：
  - 同一個 cell 裡：frame t 跟 frame t+1 是時間鄰居
  - 也可以讓 t 跟 t+2 有一點連結
  - 不同 cell 之間：沒有時間鄰居，只靠 latent 距離

**時間結構：**
```python
time_edges = [
    (frame_t_idx, frame_t+1_idx),  # 同 cell 內的時間鄰居
    (frame_t_idx, frame_t+2_idx),  # 可選：更弱的連結
    ...
]
```

### Step 4: 跑 3D TPHATE

**腳本：** `tphate_3d_pipeline.py`

**功能：**
- 如果 `tphate` 庫可用：使用真正的 TPHATE
- 如果不可用：使用 PHATE + 時間特徵作為近似

**使用方法：**
```bash
python3 tphate_3d_pipeline.py \
    --input latents_preprocessed.npz \
    --output tphate_3d_results.npz \
    --use_pca \
    --knn 10 \
    --n_components 3 \
    --seed 42
```

**輸出格式：**
```python
{
    'Z_tphate': [N_frames, 3],    # 3D TPHATE embedding
    'cell_id': [N_frames],
    'frame_in_cell': [N_frames],
    'abs_time': [N_frames],
    'time_edges': [N_edges, 2]     # 時間邊列表
}
```

## 完整流程（一鍵運行）

**腳本：** `run_full_tphate_pipeline.sh`

```bash
cd ~/ivf_repo
./run_full_tphate_pipeline.sh
```

這個腳本會自動：
1. 檢查並安裝依賴（sklearn, phate）
2. 運行 Step 1: Export latents
3. 運行 Step 2: Preprocess
4. 運行 Step 3-4: TPHATE

## Step 5: 後續分析（需要自己實現）

有了 `tphate_3d_results.npz` 之後，可以進行：

### 1. 軌跡分析
- 每個 cell 是一條在 3D 空間裡的 curve
- 計算速度（弧長 / 時間）
- 計算曲率、折返點（代表發育階段轉換）
- 比較 good outcome vs bad outcome 的軌跡形狀

### 2. TDA (Persistent Homology)
- 對所有 frame 或某一時間窗口的 3D 點雲做 Vietoris-Rips / alpha complex
- 看 Betti-0/1/2 的變化：有沒有 loop、branch、connected components 的變化

### 3. Clustering / 分亞型
- 在 3D TPHATE 空間裡用 HDBSCAN / k-means 對 cell 的整條軌跡聚類
- 找出「發育模式的亞型」，再跟 clinical label（implantation, pregnancy）對應

### 4. Feature for Downstream Classifier
- 把每個 cell 的 3D TPHATE 軌跡變成 summary feature：
  - 3 個維度各自的 mean / std / early vs late 差值
- 再丟到 logistic regression / random forest 去預測 outcome

## 依賴安裝

在 CHTC 上：
```bash
pip install --user scikit-learn phate
# 如果 tphate 可用（可選）
pip install --user tphate
```

## 參數調整建議

### knn 參數
- 小數據集（< 1000 frames）：knn=5
- 中等數據集（1000-10000 frames）：knn=10
- 大數據集（> 10000 frames）：knn=20

### PCA 維度
- 原始 latent 維度很高（256）：建議 PCA 降到 32-50 維
- 如果原始維度已經較低（< 64）：可以跳過 PCA

### 時間權重（PHATE 近似時）
- 在 `tphate_3d_pipeline.py` 中的 `time_weight = 0.1` 可以調整
- 如果時間結構很重要：增加到 0.2-0.3
- 如果更依賴 latent geometry：降低到 0.05

## 故障排除

### 問題 1: `ModuleNotFoundError: No module named 'phate'`
**解決：** `pip install --user phate`

### 問題 2: TPHATE 結果不穩定
**解決：** 
- 調整 `knn` 參數（5, 10, 20）
- 使用不同的 `random_seed`
- 檢查數據預處理是否正確（標準化、PCA）

### 問題 3: 內存不足
**解決：**
- 使用 `--max_sequences` 限制處理的序列數
- 降低 PCA 維度
- 分批處理數據

## 參考文獻

- PHATE: https://github.com/KrishnaswamyLab/PHATE
- T-PHATE: (如果可用) 查看 tphate 庫文檔
- Persistent Homology: 使用 ripser 或 gudhi 庫

