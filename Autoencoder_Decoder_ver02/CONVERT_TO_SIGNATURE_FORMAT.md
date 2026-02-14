# 轉換 Latent 數據為 Path Signature 格式

## 📋 概述

你同學的 `export_signatures.py` 腳本期望以下格式：

```
latents/
  ├── {model_name}.npy    # 所有 latent vectors，shape: [N, latent_dim]
  └── {model_name}.csv     # CSV 文件，包含 'cell_id' 列，每行對應一個 frame
```

這個腳本可以將你現有的 latent 數據轉換成這個格式。

---

## 🔍 找到你同學的 Latent 文件位置

根據 GitHub 代碼，你同學的 latent 文件應該在：

### 選項 1: 從 `extract_all_latent_trajectories.py` 生成

**位置：** `model_latents/{model_version}/latents/embryo_{cell_id}.npy`

**例子：**
```bash
model_latents/
  └── v1_baseline/
      └── latents/
          ├── embryo_RI382-2.npy
          ├── embryo_BA518-7.npy
          └── ...
```

### 選項 2: 從 `export_all_frame_latents_direct.py` 生成

**位置：** `*.npz` 文件（例如 `latents_all_frames_direct.npz`）

**包含的數據：**
- `Z`: latent vectors [N, latent_dim]
- `cell_id`: 每個 frame 對應的 cell_id
- `frame_in_cell`: frame 索引
- 等等

---

## 🚀 使用方法

### 方法 1: 從個別 .npy 文件轉換

如果你有從 `extract_all_latent_trajectories.py` 生成的個別文件：

```bash
python convert_to_signature_format.py \
    --input_type individual_npy \
    --input model_latents/v1_baseline/latents \
    --model_name v1_baseline \
    --output_dir latents
```

**輸出：**
- `latents/v1_baseline.npy`
- `latents/v1_baseline.csv`

### 方法 2: 從 .npz 文件轉換

如果你有從 `export_all_frame_latents_direct.py` 生成的 .npz 文件：

```bash
python convert_to_signature_format.py \
    --input_type npz \
    --input tphate_results_final/latents_all_frames_direct.npz \
    --model_name epoch50_direct \
    --output_dir latents
```

**輸出：**
- `latents/epoch50_direct.npy`
- `latents/epoch50_direct.csv`

---

## ✅ 驗證輸出

轉換完成後，檢查文件：

```python
import numpy as np
import pandas as pd

# 檢查 .npy 文件
Z = np.load("latents/v1_baseline.npy")
print(f"Latent array shape: {Z.shape}")  # 應該是 [N, latent_dim]

# 檢查 .csv 文件
df = pd.read_csv("latents/v1_baseline.csv")
print(f"CSV shape: {df.shape}")  # 應該是 [N, 1]
print(f"Columns: {df.columns}")  # 應該有 'cell_id'
print(f"Unique embryos: {df['cell_id'].nunique()}")

# 驗證一致性
assert len(df) == Z.shape[0], "CSV 和 .npy 的行數應該一致！"
```

---

## 🎯 使用轉換後的文件

現在你可以使用你同學的 `export_signatures.py` 腳本：

```bash
python export_signatures.py --name v1_baseline
```

這會：
1. 讀取 `latents/v1_baseline.npy` 和 `latents/v1_baseline.csv`
2. 對每個胚胎計算 Path Signature
3. 保存到 `signatures/v1_baseline_sigs.csv`

---

## 📝 常見問題

### Q: 我的 latent 文件在哪裡？

**A:** 檢查以下位置：

1. **CHTC 上的結果：**
   ```bash
   /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/
   ```

2. **本地結果：**
   ```bash
   model_latents/v1_baseline/latents/
   # 或
   tphate_results_final/latents_all_frames_direct.npz
   ```

### Q: 文件名格式不匹配怎麼辦？

**A:** 腳本會自動嘗試多種文件名模式：
- `embryo_{cell_id}.npy`
- `{cell_id}_z.npy`
- 如果都不匹配，會使用文件名（去掉擴展名）

### Q: 轉換後的文件太大怎麼辦？

**A:** Path Signature 計算可能需要較多內存。如果文件太大，可以：
1. 只轉換部分胚胎（修改輸入目錄）
2. 使用 `--max_embryos` 限制處理數量

---

## 🔗 相關文件

- `export_signatures.py` - 你同學的 Path Signature 計算腳本
- `extract_all_latent_trajectories.py` - 你的 latent 提取腳本
- `export_all_frame_latents_direct.py` - 你的直接 frame 提取腳本






