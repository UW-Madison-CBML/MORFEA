# 直接從 tar.gz 讀取（無需解壓）

## 問題

解壓時遇到 "Disk quota exceeded" 錯誤，即使設定了 `TMPDIR` 到 staging。

## 解決方案

`analyze_trajectory_curvature.py` 現在支援直接從 tar.gz 讀取 frames，**不需要解壓**！

## 使用方法

### 方法 1: 讓腳本自動偵測（推薦）

腳本會自動嘗試以下順序：
1. 指定的 `--data_root` 目錄
2. 常見的 staging 目錄
3. **如果都找不到，自動使用 group tar.gz**

```bash
cd /staging/groups/bhaskar_group/rho9/ivf_analysis

# 不需要指定 data_root，腳本會自動找到 tar.gz
python3 scripts/analyze_trajectory_curvature.py --video_name ZS435-5
python3 scripts/analyze_trajectory_curvature.py --video_name RS363-7
```

### 方法 2: 明確指定 tar.gz 路徑

```bash
python3 scripts/analyze_trajectory_curvature.py \
    --video_name ZS435-5 \
    --data_root /staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz
```

## 優點

✅ **不需要解壓** - 直接從 tar.gz 讀取，節省空間
✅ **自動 fallback** - 如果找不到目錄，自動使用 tar.gz
✅ **無 quota 問題** - 不需要寫入任何檔案到磁碟

## 工作原理

1. 腳本首先嘗試從已解壓的目錄讀取
2. 如果找不到，自動檢查 group tar.gz (`/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz`)
3. 使用 Python 的 `tarfile` 模組直接從 tar.gz 讀取 frames
4. 在記憶體中處理，不寫入磁碟

## 注意事項

- 從 tar.gz 讀取會比從目錄讀取稍慢（因為需要解壓縮）
- 但對於分析任務來說，這個速度差異通常可以接受
- 如果有多個 cells 需要分析，建議還是解壓（但現在至少可以運行了）

## 如果還是遇到問題

如果腳本還是找不到資料，可以手動檢查：

```bash
# 檢查 tar.gz 是否存在
ls -lh /staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz

# 檢查特定 cell 是否存在
tar -tzf /staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz | \
    grep "^embryo_dataset/ZS435-5/" | head -5
```

