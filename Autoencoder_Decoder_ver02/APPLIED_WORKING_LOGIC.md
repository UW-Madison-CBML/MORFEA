# 套用成功方法的邏輯到 CHTC 提取腳本

## 改變摘要

我們把之前成功的方法（`export_all_frame_latents_direct.py`）的邏輯套用到 CHTC 的提取腳本中。

### 關鍵改變：Device 自動檢測

**之前的問題：**
- 依賴 bash 變數 `$DEVICE` 傳遞給 Python
- bash 變數可能為空或錯誤
- 導致 `--device` 參數無效錯誤

**現在的解決方案（套用成功方法）：**
- ✅ **在 Python 內部自動檢測 device**（就像 `export_all_frame_latents_direct.py` 一樣）
- ✅ 不需要 bash 變數
- ✅ 如果 `--device` 未指定或無效，自動使用 `torch.cuda.is_available()` 檢測

## 修改的文件

### 1. `extract_all_latent_trajectories.py`

**改變：**
```python
# 之前：從 argparse 直接使用 device，默認是 "cpu"
device=args.device  # 如果為空會出錯

# 現在：自動檢測（套用成功方法的邏輯）
if args.device and args.device in ["cpu", "cuda"]:
    device = args.device
else:
    # 如果 device 參數無效或未指定，自動檢測
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚠️  Device parameter invalid or missing, auto-detected: {device}")

# 如果指定了 cuda 但實際不可用，降級到 cpu
if device == "cuda" and not torch.cuda.is_available():
    print(f"⚠️  CUDA requested but not available, falling back to CPU")
    device = "cpu"
```

**argparse 改變：**
```python
# 之前：
default="cpu"

# 現在：
default=None  # 讓程序自動檢測
```

### 2. `extract_latents.sh`

**改變：**
- ✅ 移除了 `--device "$DEVICE"` 參數
- ✅ 保留了 GPU 檢測（僅用於日誌顯示）
- ✅ Python 腳本會自動處理 device

```bash
# 之前：
DEVICE=$(python3 -c "...")
python3 extract_all_latent_trajectories.py --device "$DEVICE" ...

# 現在：
# 只顯示 GPU 信息（用於日誌），不傳遞參數
python3 << 'PYTHON_EOF'
import torch
if torch.cuda.is_available():
    print("GPU detected via PyTorch")
else:
    print("No GPU detected")
PYTHON_EOF

# Python 腳本會自動檢測
python3 extract_all_latent_trajectories.py ...  # 不需要 --device
```

## 為什麼這個方法有效？

### 比喻說明

**之前的方法（容易出錯）：**
就像兩個人傳話，第一個人在 bash 中說"用 GPU"，但傳話時可能出錯或沒說清楚，第二個人（Python）收到空消息就不知道該怎麼辦。

**現在的方法（成功方法）：**
就像 Python 自己有眼睛和耳朵，能直接看到有沒有 GPU，不需要別人告訴它。這就是為什麼 `export_all_frame_latents_direct.py` 能成功的原因 - 它直接在 Python 內部處理，不需要依賴外部的 bash 變數。

## 使用方式

現在腳本的使用方式更簡單了：

```bash
# 不需要指定 --device，會自動檢測
python3 extract_all_latent_trajectories.py \
    --checkpoint checkpoint.pt \
    --model_version v1_baseline \
    --index_csv index.csv
```

或者如果需要強制指定：

```bash
# 也可以手動指定（但通常不需要）
python3 extract_all_latent_trajectories.py \
    --checkpoint checkpoint.pt \
    --model_version v1_baseline \
    --device cuda  # 如果指定，會驗證；如果不可用會降級到 cpu
```

## 下一步

1. 上傳修改後的文件到 CHTC
2. 重新提交任務
3. 應該不會再出現 `--device` 參數錯誤了








