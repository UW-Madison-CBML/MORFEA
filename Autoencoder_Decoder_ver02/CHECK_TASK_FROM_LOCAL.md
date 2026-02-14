# 從本地檢查 CHTC 任務狀態

## 重新連接到 CHTC

```bash
ssh rho9@ap2001.chtc.wisc.edu
```

## 檢查任務狀態

```bash
# 1. 查看任務是否還在運行
condor_q -submitter rho9

# 2. 查看最新輸出
condor_tail 2852953.0 | tail -20

# 3. 查看任務運行時間
condor_q -long 2852953.0 | grep -E 'JobCurrentStartDate|RemoteWallClockTime'

# 4. 檢查是否有結果（即使部分完成）
ls -lh /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/ 2>/dev/null || echo '目錄不存在'
```

## 從最後的輸出看

您看到的最後輸出是：
```
Loading model from checkpoint
```

這表示任務可能：
- 正在加載模型（這可能需要一些時間）
- 或者已經完成加載，正在處理數據

## 建議

1. **重新 SSH 到 CHTC** 檢查任務狀態
2. 如果任務還在運行，可以繼續等待
3. 如果任務已經完成或失敗，檢查結果目錄

