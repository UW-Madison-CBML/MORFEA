# 使用 Group 層級的資料

## 目前狀況

✅ **你的個人資料已清理**
- `/staging/groups/bhaskar_group/rho9/` 只剩 44K
- 12GB 資料已刪除

✅ **Group 層級有資料**
- `/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz` (12GB)
- 擁有者: `jlundsgaard`
- 目前只有 tar.gz，沒有解壓目錄

## 使用方式

### 方案 1: 只解壓需要的 cells (推薦)

使用 `extract_from_group_tar.sh` 腳本，只解壓你需要的 cells：

```bash
# 在 CHTC 上執行
cd /staging/groups/bhaskar_group/rho9/ivf_analysis

# 上傳腳本（如果還沒有的話）
# 從本地: scp scripts/extract_from_group_tar.sh rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/ivf_analysis/

# 執行腳本（預設解壓 ZS435-5 和 RS363-7）
bash scripts/extract_from_group_tar.sh

# 或指定其他 cells
bash scripts/extract_from_group_tar.sh ZS435-5 RS363-7 RI382-2
```

**優點:**
- 只解壓需要的 cells，節省空間
- 解壓到你的個人 staging 目錄，不會影響其他人
- 腳本會自動設定 `TMPDIR`，避免使用 home directory

### 方案 2: 檢查 tar.gz 內容

在解壓前，可以先檢查 tar.gz 裡面有哪些 cells：

```bash
# 列出所有 cells
tar -tzf /staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz | grep "^embryo_dataset/" | cut -d'/' -f2 | sort -u

# 檢查特定 cell 是否存在
tar -tzf /staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz | grep "^embryo_dataset/ZS435-5/" | head -5
```

### 方案 3: 直接使用（如果腳本支援）

如果 `analyze_trajectory_curvature.py` 或其他腳本支援直接從 tar.gz 讀取，就不需要解壓。但目前這些腳本都需要已解壓的目錄。

## 解壓後的資料位置

解壓後的資料會放在：
```
/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset/
├── ZS435-5/
│   ├── frame1.jpeg
│   ├── frame2.jpeg
│   └── ...
└── RS363-7/
    ├── frame1.jpeg
    └── ...
```

## 使用解壓的資料

`analyze_trajectory_curvature.py` 會自動尋找以下路徑：
1. 指定的 `--data_root` 參數
2. `/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset` (你的個人 staging)
3. `/staging/groups/bhaskar_group/ivf/embryo_dataset` (group 層級，如果有的話)

所以解壓後，直接執行：

```bash
cd /staging/groups/bhaskar_group/rho9/ivf_analysis
python3 scripts/analyze_trajectory_curvature.py --video_name ZS435-5
python3 scripts/analyze_trajectory_curvature.py --video_name RS363-7
```

腳本會自動找到解壓的資料。

## 注意事項

1. **空間限制**: 雖然 staging 空間很大，但還是建議只解壓需要的 cells
2. **TMPDIR**: 腳本會自動設定 `TMPDIR` 到 staging，避免使用 home directory
3. **權限**: Group tar.gz 屬於 `jlundsgaard`，但應該可以讀取（因為在 group 目錄下）

## 如果遇到問題

1. **"Disk quota exceeded"**: 
   - 確認 `TMPDIR` 已設定到 staging
   - 檢查是否在解壓到 home directory

2. **"Cell directory not found"**:
   - 確認 cell 名稱正確（大小寫敏感）
   - 檢查 tar.gz 內容確認 cell 是否存在

3. **解壓不完整**:
   - 檢查解壓過程是否有錯誤訊息
   - 確認 staging 空間足夠

