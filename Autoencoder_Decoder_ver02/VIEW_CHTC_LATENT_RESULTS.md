# 📊 CHTC Latent Vector Extraction 結果查看指南

## 🔍 檢查 CHTC 上的結果

由於無法直接自動連接，請在 CHTC 上手動執行以下命令：

### 1. 連接到 CHTC
```bash
ssh rho9@ap2001.chtc.wisc.edu
```

### 2. 檢查結果目錄
```bash
# 檢查目錄是否存在
ls -lh /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/

# 檢查 latents 子目錄
ls -lh /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/
```

### 3. 統計檔案數量
```bash
# 計算有多少個 latent 檔案
ls -1 /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/*.npy 2>/dev/null | wc -l

# 查看檔案列表（最新的10個）
ls -lht /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/*.npy 2>/dev/null | head -10

# 查看總大小
du -sh /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/
```

### 4. 檢查 metadata.json
```bash
# 查看 metadata 內容
cat /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/metadata.json

# 或使用 Python 格式化查看
python3 -c "import json; d=json.load(open('/staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/metadata.json')); print(json.dumps(d, indent=2))"
```

### 5. 查看任務日誌
```bash
# 查看輸出日誌（最後50行）
tail -50 ~/logs/extract_latents_v1_baseline.out

# 查看錯誤日誌
tail -50 ~/logs/extract_latents_v1_baseline.err

# 查看 condor 日誌
tail -100 ~/logs/extract_latents_v1_baseline.log
```

### 6. 查看任務狀態
```bash
# 檢查任務是否還在運行
condor_q -submitter rho9 | grep extract

# 查看任務歷史
condor_history -limit 5 -submitter rho9 | grep extract
```

---

## 📥 下載結果到本地

### 方法 1: 下載所有 latent 檔案
```bash
# 在本地終端執行
mkdir -p "Code/Autoencoder_Decoder_ver02/model_latents/v1_baseline/latents"

scp -r rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/*.npy \
    "Code/Autoencoder_Decoder_ver02/model_latents/v1_baseline/latents/"
```

### 方法 2: 下載整個目錄（包括 metadata）
```bash
# 在本地終端執行
scp -r rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/model_latents/v1_baseline \
    "Code/Autoencoder_Decoder_ver02/model_latents/"
```

### 方法 3: 使用 rsync（推薦，可以續傳）
```bash
# 在本地終端執行
rsync -avz --progress \
    rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/ \
    "Code/Autoencoder_Decoder_ver02/model_latents/v1_baseline/"
```

---

## 📋 檢查下載結果

下載完成後，在本地執行：

```bash
cd "Code/Autoencoder_Decoder_ver02"

# 檢查檔案數量
ls -1 model_latents/v1_baseline/latents/*.npy | wc -l

# 查看檔案列表
ls -lh model_latents/v1_baseline/latents/*.npy | head -10

# 查看 metadata
cat model_latents/v1_baseline/metadata.json

# 使用 Python 查看詳細資訊
python3 << 'EOF'
import numpy as np
import json
from pathlib import Path

base_dir = Path("model_latents/v1_baseline")

# 檢查 metadata
if (base_dir / "metadata.json").exists():
    with open(base_dir / "metadata.json") as f:
        meta = json.load(f)
    print("Metadata:")
    print(json.dumps(meta, indent=2))

# 檢查 latent 檔案
latent_files = list((base_dir / "latents").glob("*.npy"))
print(f"\n找到 {len(latent_files)} 個 latent 檔案")

for f in sorted(latent_files)[:5]:
    data = np.load(f)
    print(f"  - {f.name}: shape={data.shape}, dtype={data.dtype}")
EOF
```

---

## 💡 提示

1. **如果連接超時**：可能是網路問題或 CHTC 暫時無法訪問，稍後再試
2. **如果目錄不存在**：任務可能還在運行或尚未開始，檢查 `condor_q` 查看任務狀態
3. **如果檔案很少**：任務可能還在進行中，可以查看日誌了解進度

