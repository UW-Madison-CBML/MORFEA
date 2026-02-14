# ✅ 修復 index.csv 自動建立問題

## 問題根源

訓練任務失敗因為：
- `index.csv` 沒有被建立
- `run_train.sh` 中的 `build_index.py` 因為資料路徑問題失敗
- `dataset_ivf.py` 嘗試讀取不存在的 `index.csv` 時直接失敗

## 解決方案

修改 `dataset_ivf.py`，讓它在找不到 `index.csv` 時自動建立。

### 修改內容

1. **添加 build_index 導入**：
   ```python
   try:
       import build_index
   except ImportError:
       build_index = None
   ```

2. **在 `__init__` 中自動建立 index.csv**：
   - 檢查 `index.csv` 是否存在
   - 如果不存在，調用 `build_index.main()` 建立
   - 建立後再次檢查，確保成功
   - 最後才讀取 CSV

### 優勢

- ✅ 在 GPU 節點上執行，資料路徑一定可用
- ✅ 邏輯集中：唯一使用 index 的地方負責建立
- ✅ 自動化：不需要手動步驟
- ✅ 清晰的錯誤訊息：如果建立失敗會給出明確提示

## 已確認

- ✅ `train_h200_lab.sub` 包含 `build_index.py` 在 `transfer_input_files`
- ✅ `dataset_ivf.py` 已修改為自動建立 `index.csv`
- ✅ 錯誤處理已完善

## 下一步

重新提交訓練任務：

```bash
cd ~/ivf/Raffael/2025-11-19
condor_submit train_h200_lab.sub
condor_tail -f <JOB_ID>.0
```

預期輸出中會看到：
```
[dataset_ivf] index.csv not found, building it with build_index.py...
Found XXX cell directories in data
Processing cells: 100%|██████████| ...
✓ Wrote index.csv with YYY sequences
[dataset_ivf] ✓ Successfully created index.csv
[dataset_ivf] Loaded index with YYY rows
```

然後訓練應該會正常開始！

