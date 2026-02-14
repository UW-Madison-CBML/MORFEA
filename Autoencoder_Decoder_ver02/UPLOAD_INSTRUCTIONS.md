# 上傳文件到 CHTC 的說明

## 如果 scp 卡住或等待輸入

`scp` 命令會提示您輸入 SSH 密碼。如果您已經設置了 SSH 密鑰，應該會自動認證；否則需要手動輸入密碼。

## 手動上傳（如果腳本卡住）

如果腳本卡住，您可以直接在終端中執行以下命令：

```bash
cd "/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

# 1. 上傳 extract_all_latent_trajectories.py
scp extract_all_latent_trajectories.py rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/

# 2. 上傳 extract_latents.sh
scp extract_latents.sh rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/
```

## 驗證上傳

上傳完成後，SSH 到 CHTC 並確認文件已更新：

```bash
ssh rho9@ap2001.chtc.wisc.edu

# 檢查文件時間戳（應該是最新的）
ls -lh /staging/groups/bhaskar_group/rho9/extract_all_latent_trajectories.py
ls -lh /staging/groups/bhaskar_group/rho9/extract_latents.sh

# 確認修改內容（查看 device 自動檢測的代碼）
grep -A 5 "Auto-detect device" /staging/groups/bhaskar_group/rho9/extract_all_latent_trajectories.py
```

## 重新提交任務

文件上傳完成後，重新提交任務：

```bash
# 在 CHTC 上
condor_submit ~/extract_latents_from_home.sub
```

## 主要改變

1. **extract_all_latent_trajectories.py**:
   - 現在會自動檢測 device（不需要 bash 變數）
   - 如果 `--device` 未指定，會自動使用 `torch.cuda.is_available()` 檢測

2. **extract_latents.sh**:
   - 移除了 `--device "$DEVICE"` 參數
   - Python 腳本會自動處理 device 檢測

這樣就解決了之前 `--device` 參數為空的問題。








