# ✅ CHTC 訓練準備完成

## 📦 已準備的文件

所有必要的文件已經準備完成：

- ✅ `train.py` - 訓練腳本
- ✅ `model.py` - ConvLSTM Autoencoder 模型（剛創建）
- ✅ `losses.py` - 損失函數
- ✅ `dataset_ivf.py` - 資料集載入器（已修復圖片損壞問題）
- ✅ `build_index.py` - 建立資料索引
- ✅ `run_train.sh` - 訓練執行腳本
- ✅ `train_h200_lab.sub` - CHTC 提交文件（已更新，移除 conv_lstm.py）

## 🚀 在 CHTC 上運行的步驟

### 方法 1: 從 GitHub Clone（推薦）

如果你已經將文件上傳到 GitHub：

```bash
# 1. SSH 到 CHTC
ssh rho9@ap2001.chtc.wisc.edu

# 2. Clone repository
cd ~
git clone https://github.com/UW-Madison-CBML/ivf.git
cd ivf/Raffael/2025-11-19  # 或你的實際路徑

# 3. 確認文件存在
ls -lh

# 4. 設置權限
chmod +x run_train.sh
mkdir -p logs

# 5. 提交任務
condor_submit train_h200_lab.sub
```

### 方法 2: 直接上傳文件到 CHTC

如果還沒有上傳到 GitHub，可以直接上傳文件：

```bash
# 在你的 Mac 上，使用 scp 上傳整個目錄
cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"
scp -r train.py model.py losses.py dataset_ivf.py build_index.py run_train.sh train_h200_lab.sub rho9@ap2001.chtc.wisc.edu:~/ivf_training/

# 然後 SSH 到 CHTC
ssh rho9@ap2001.chtc.wisc.edu
cd ~/ivf_training
chmod +x run_train.sh
mkdir -p logs
condor_submit train_h200_lab.sub
```

## 📊 監控訓練

```bash
# 查看任務狀態
condor_q

# 查看實時輸出（替換 <cluster>.<proc> 為實際 job ID）
condor_tail <cluster>.<proc>

# 查看錯誤
condor_tail -stderr <cluster>.<proc>
```

## ⚠️ 重要提醒

1. **model.py 是新創建的**：確保這個文件已經上傳到 CHTC 或 GitHub
2. **dataset_ivf.py 已修復**：現在可以處理損壞的圖片文件
3. **不需要 conv_lstm.py**：已從提交文件中移除
4. **資料會在 GPU 節點上自動建立**：`run_train.sh` 會自動執行 `build_index.py`

## 🎯 訓練配置

- **GPU**: H200 (1 GPU)
- **Epochs**: 50
- **Batch Size**: 8
- **Sequence Length**: 20
- **Learning Rate**: 3e-4
- **Checkpoint**: 每 5 個 epoch 保存一次

訓練完成後，結果會保存在 `checkpoints/` 目錄中。

