# 🚀 從零開始：CHTC 訓練完整流程

## 📋 完整步驟（按順序執行）

### 步驟 1: SSH 到 CHTC

在你的 Mac 終端執行：

```bash
ssh rho9@ap2001.chtc.wisc.edu
```

輸入密碼和 Duo 驗證碼（6位數字）

---

### 步驟 2: Clone Repository

```bash
cd ~
git clone https://github.com/UW-Madison-CBML/ivf.git
cd ivf/Raffael/2025-11-19
```

---

### 步驟 3: 確認文件存在

```bash
ls -lh
```

應該看到這些文件：
- ✅ `model.py`
- ✅ `losses.py`
- ✅ `train.py`
- ✅ `dataset_ivf.py`
- ✅ `build_index.py`
- ✅ `run_train.sh`
- ✅ `train_h200_lab.sub`

---

### 步驟 4: 設置權限

```bash
chmod +x run_train.sh
mkdir -p logs
```

---

### 步驟 5: 提交訓練任務

```bash
condor_submit train_h200_lab.sub
```

你會看到類似這樣的輸出：
```
Submitting job(s).
1 job(s) submitted to cluster 2609123.
```

**記下這個 cluster 號碼！**

---

### 步驟 6: 查看任務狀態

```bash
condor_q
```

狀態說明：
- **I** = Idle（等待 GPU）
- **R** = Running（正在訓練）
- **H** = Held（有錯誤）

---

### 步驟 7: 監控訓練（任務運行時）

```bash
# 查看實時輸出（替換 2609123.0 為你的實際 job ID）
condor_tail 2609123.0

# 查看錯誤（如果有）
condor_tail -stderr 2609123.0
```

---

### 步驟 8: 檢查結果（訓練完成後）

```bash
# 結果保存在 staging
ls -lh /staging/groups/bhaskar_group/ivf/results/

# 查看本地日誌
tail -n 100 logs/train_*.out
```

---

## ⚡ 快速一鍵執行（複製全部）

如果你想一次性執行步驟 2-6：

```bash
cd ~ && \
git clone https://github.com/UW-Madison-CBML/ivf.git && \
cd ivf/Raffael/2025-11-19 && \
chmod +x run_train.sh && \
mkdir -p logs && \
condor_submit train_h200_lab.sub && \
condor_q
```

---

## ❓ 常見問題

### Q: 任務一直顯示 Idle？

```bash
# 查看詳細原因
condor_q -better-analyze <cluster.proc>
```

可能原因：
- H200 GPU 暫時不可用（需要等待）
- 實驗室配額問題（已配置，通常不是問題）

### Q: 任務失敗了？

```bash
# 查看錯誤日誌
condor_tail -stderr <cluster.proc>
```

### Q: 如何取消任務？

```bash
condor_rm <cluster.proc>
```

---

## 📊 訓練預期時間

- **每個 epoch**: 約 10-30 分鐘
- **總共 50 epochs**: 約 8-25 小時
- **Checkpoint**: 每 5 個 epoch 保存一次

---

## ✅ 訓練完成後

結果會自動保存到：
```
/staging/groups/bhaskar_group/ivf/results/results_<cluster>_<proc>.tgz
```

解壓縮查看：
```bash
cd /staging/groups/bhaskar_group/ivf/results/
tar -xzf results_<cluster>_<proc>.tgz
```

---

## 🎯 下一步

訓練完成後，你可以：
1. 載入模型 checkpoint
2. 提取 `z_seq` 和 `z_last` latent vectors
3. 進行 T-PHATE 或 TDA 分析


