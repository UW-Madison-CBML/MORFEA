# 上傳完成後的下一步操作

## ✅ 文件已上傳

現在需要在 CHTC 上執行以下步驟：

## 步驟 1: 驗證文件已上傳

在 CHTC 上執行：
```bash
# 檢查 staging 上的腳本
ls -lh /staging/groups/bhaskar_group/rho9/extract_latents.sh

# 檢查 home 目錄的 submit 文件
ls -lh ~/extract_latents_from_home.sub

# 確認修復已包含
grep -A 3 'if \[ -z "\$DEVICE" \]' /staging/groups/bhaskar_group/rho9/extract_latents.sh
grep 'Pre-creating output directory' /staging/groups/bhaskar_group/rho9/extract_latents.sh
grep 'MaxJobRuntime' ~/extract_latents_from_home.sub
```

## 步驟 2: 移除舊任務（如果還在）

```bash
# 查看當前任務
condor_q -submitter rho9

# 如果有舊任務（如 2851275.0），移除它
condor_rm 2851275.0
```

## 步驟 3: 檢查並設置 submit 文件

```bash
# 查看 submit 文件最後幾行
tail -10 ~/extract_latents_from_home.sub
```

如果最後沒有 `checkpoint`、`model_version` 和 `queue`，需要添加：

```bash
# 檢查 checkpoint 是否存在
ls -lh /staging/groups/bhaskar_group/rho9/checkpoints/checkpoint_epoch_50.pt

# 如果存在，添加到 submit 文件末尾
echo "" >> ~/extract_latents_from_home.sub
echo "checkpoint = /staging/groups/bhaskar_group/rho9/checkpoints/checkpoint_epoch_50.pt" >> ~/extract_latents_from_home.sub
echo "model_version = v1_baseline" >> ~/extract_latents_from_home.sub
echo "queue" >> ~/extract_latents_from_home.sub
```

## 步驟 4: 提交任務

```bash
condor_submit ~/extract_latents_from_home.sub
```

這會返回一個新的 job ID，記下它。

## 步驟 5: 監控任務

```bash
# 查看任務狀態
condor_q -submitter rho9

# 查看任務詳細信息（當任務在 IDLE 時）
condor_q -better-analyze <job_id>

# 實時查看輸出（當任務開始運行後）
condor_tail -f <job_id>

# 或者查看日誌文件
tail -f ~/logs/extract_latents_v1_baseline.out
```

## 步驟 6: 檢查結果（任務完成後）

```bash
# 檢查結果目錄
ls -lh /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/

# 統計提取的文件數量
ls -1 /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents/*.npy 2>/dev/null | wc -l

# 查看 metadata
cat /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/metadata.json
```

## 預期改進

修復後的任務應該：
- ✅ DEVICE 變量不會為空（有默認值 fallback）
- ✅ 預先創建目錄結構（方便調試）
- ✅ 運行時間限制為 24 小時（避免中途被終止）
- ✅ 更詳細的結果檢查和報告

