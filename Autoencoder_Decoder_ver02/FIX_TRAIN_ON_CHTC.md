# 在 CHTC 上直接修復 train.py

如果 `git pull` 沒有更新，直接在 CHTC 上執行以下命令：

## 方法 1: 使用 sed 直接插入（最簡單）

```bash
cd ~/ivf/Raffael/2025-11-19

# 備份
cp train.py train.py.backup

# 找到要插入的位置（在 "train_dataset = IVFSequenceDataset" 之前）
# 然後用 sed 插入檢查代碼

# 先找到行號
LINE_NUM=$(grep -n "train_dataset = IVFSequenceDataset" train.py | head -1 | cut -d: -f1)

# 在該行之前插入檢查代碼
sed -i "${LINE_NUM}i\\
    # ★ Ensure index.csv exists before loading dataset ★\\
    index_path = Path(index_csv)\\
    if not index_path.exists():\\
        print(f\"[train] {index_csv} not found, building with build_index.py ...\", flush=True)\\
        if build_index is None:\\
            raise ImportError(\\
                f\"[train] build_index module not available. \"\\
                \"Cannot auto-generate index.csv. Please run build_index.py manually.\"\\
            )\\
        build_index.main()\\
        if not index_path.exists():\\
            raise FileNotFoundError(\\
                f\"[train] After running build_index.main(), still no {index_csv}. \"\\
                \"Check that symlink 'data' -> /project/bhaskar_group/ivf has valid content.\"\\
            )\\
        print(f\"[train] ✓ Successfully created {index_csv}\", flush=True)\\
    \\
" train.py

# 檢查語法
python3 -m py_compile train.py && echo "✓ OK" || echo "✗ Error"
```

## 方法 2: 直接從 GitHub 下載最新版本（最可靠）

```bash
cd ~/ivf/Raffael/2025-11-19

# 備份
cp train.py train.py.backup

# 從 GitHub 下載最新版本
curl -L -o train.py "https://raw.githubusercontent.com/UW-Madison-CBML/ivf/main/Raffael/2025-11-19/train.py" || \
curl -L -o train.py "https://raw.githubusercontent.com/Grnho/ivf/main/Code/Autoencoder_Decoder_ver02/train.py"

# 檢查語法
python3 -m py_compile train.py && echo "✓ OK" || echo "✗ Error"

# 驗證更新
grep "Ensure index.csv exists" train.py && echo "✓ Update successful!" || echo "✗ Update failed"
```

## 方法 3: 手動編輯（如果上面都不行）

```bash
cd ~/ivf/Raffael/2025-11-19
nano train.py
```

找到這一行（大約在第 95 行）：
```python
    train_dataset = IVFSequenceDataset(index_csv, resize=128, norm="minmax01")
```

在這一行**之前**插入以下代碼：

```python
    # ★ Ensure index.csv exists before loading dataset ★
    index_path = Path(index_csv)
    if not index_path.exists():
        print(f"[train] {index_csv} not found, building with build_index.py ...", flush=True)
        if build_index is None:
            raise ImportError(
                f"[train] build_index module not available. "
                "Cannot auto-generate index.csv. Please run build_index.py manually."
            )
        build_index.main()
        if not index_path.exists():
            raise FileNotFoundError(
                f"[train] After running build_index.main(), still no {index_csv}. "
                "Check that symlink 'data' -> /project/bhaskar_group/ivf has valid content."
            )
        print(f"[train] ✓ Successfully created {index_csv}", flush=True)
    
```

然後：
- `Ctrl + O` → 存檔
- `Enter`
- `Ctrl + X` → 離開

## 驗證更新

```bash
# 檢查是否包含新代碼
grep -A 5 "Ensure index.csv exists" train.py

# 檢查語法
python3 -m py_compile train.py

# 如果都 OK，提交新 job
condor_submit train_h200_lab.sub
```





