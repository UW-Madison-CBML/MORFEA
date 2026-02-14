# Latent Extraction 問題修復總結

## 發現的問題

1. **DEVICE 變量為空**：導致 Python 腳本無法啟動
2. **任務運行時間超過12小時被終止**：任務被 evicted
3. **目錄創建問題**：如果 Python 腳本失敗，目錄不會被創建

## 已實施的修復

### 1. 修復 DEVICE 變量問題
- 添加檢查確保 DEVICE 永遠不會是空字串
- 如果檢測失敗，默認使用 `cuda`（因為任務請求了 GPU）

### 2. 增加運行時間限制
- 在 `extract_latents_from_home.sub` 中添加 `+MaxJobRuntime = 86400`（24小時）
- 防止任務中途被終止

### 3. 預先創建輸出目錄結構
- 在運行 Python 腳本之前，先創建目錄結構
- 這樣即使 Python 腳本失敗，我們也能看到目錄是否存在
- 有助於調試和確認目錄創建是否有問題

### 4. 改進結果檢查
- 添加更詳細的結果檢查和報告
- 顯示找到的文件數量
- 檢查 metadata.json 是否存在

## 目錄創建邏輯說明

當前流程：
1. Bash 腳本預先創建目錄結構（新添加）
2. Python 腳本使用相對路徑 `model_latents` 作為輸出目錄
3. 目錄創建在任務工作目錄中
4. 任務結束時，`transfer_output_remaps` 將 `model_latents` 映射到 staging 目錄

**為什麼不直接寫入 staging？**
- 任務工作目錄有寫入權限，更容易調試
- `transfer_output_remaps` 在任務正常結束時會自動處理
- 如果任務失敗或被 evicted，結果可能不會被轉移（這是預期的）

## 下一步

1. 上傳修復後的文件到 CHTC
2. 重新提交任務
3. 監控任務進度
4. 確認目錄創建和文件保存

