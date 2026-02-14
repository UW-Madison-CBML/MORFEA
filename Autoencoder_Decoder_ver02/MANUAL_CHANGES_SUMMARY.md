# 手動修改說明（如果無法上傳文件）

如果 SSH 連接一直超時，無法上傳文件，您可以在 CHTC 上手動應用這些修改。

## 已完成的代碼修改摘要

### 1. extract_all_latent_trajectories.py 的修改

**位置：** `main()` 函數中，在 `args = parser.parse_args()` 之後

**添加的代碼：**
```python
# Auto-detect device (套用上次成功方法的邏輯)
# 就像 export_all_frame_latents_direct.py 一樣，直接在 Python 中處理 device
if args.device and args.device in ["cpu", "cuda"]:
    device = args.device
else:
    # 如果 device 參數無效或未指定，自動檢測（就像成功的方法一樣）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚠️  Device parameter invalid or missing, auto-detected: {device}")

# 如果指定了 cuda 但實際不可用，降級到 cpu
if device == "cuda" and not torch.cuda.is_available():
    print(f"⚠️  CUDA requested but not available, falling back to CPU")
    device = "cpu"

print(f"Using device: {device}")
```

**修改 argparse：**
```python
parser.add_argument(
    "--device",
    type=str,
    default=None,  # 改為 None，讓程序自動檢測
    help="Device to run on (default: auto-detect: cuda if available, else cpu). Valid values: 'cpu' or 'cuda'"
)
```

**修改 extract_all_trajectories 調用：**
```python
extract_all_trajectories(
    checkpoint_path=str(checkpoint_path),
    model_version_name=args.model_version,
    index_csv=args.index_csv,
    data_root=args.data_root,
    device=device,  # 使用自動檢測的 device，而不是 args.device
    batch_size=args.batch_size,
    max_embryos=args.max_embryos,
    output_base_dir=args.output_dir
)
```

### 2. extract_latents.sh 的修改

**移除的部分：**
- 移除 `DEVICE` 變數設置和檢查
- 移除所有 `--device "$DEVICE"` 參數

**修改為：**
```bash
# Detect GPU for logging (但讓 Python 腳本自動處理 device，套用成功方法的邏輯)
echo "Detecting GPU (for logging, Python will auto-detect)..."
python3 << 'PYTHON_EOF'
import torch
if torch.cuda.is_available():
    print("GPU detected via PyTorch")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("  (Python script will auto-detect and use CUDA)")
else:
    print("No GPU detected")
    print("  (Python script will auto-detect and use CPU)")
PYTHON_EOF
```

**在 python3 命令中：**
```bash
# 移除 --device "$DEVICE" 參數
python3 -u extract_all_latent_trajectories.py \
    --checkpoint "$CHECKPOINT" \
    --model_version "$MODEL_VERSION" \
    --index_csv index.csv \
    --max_embryos "$MAX_EMBRYOS" \
    --output_dir "$OUTPUT_BASE" 2>&1 | tee -a extraction_progress.log
```

## 在 CHTC 上手動修改的步驟

如果網絡連接恢復，您可以：

1. **SSH 到 CHTC**
2. **編輯文件**：
   ```bash
   cd /staging/groups/bhaskar_group/rho9
   vi extract_all_latent_trajectories.py  # 或使用 nano
   vi extract_latents.sh
   ```
3. **按照上面的修改說明進行編輯**

或者，**一旦網絡連接恢復，直接上傳文件會更簡單**。

## 當前狀態

✅ **代碼修改已完成**（在本地）
⏳ **等待網絡連接恢復以上傳文件**
✅ **一旦上傳，任務應該能正常運行**








