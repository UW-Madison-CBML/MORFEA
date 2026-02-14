# 在 CHTC 上快速修復文件

如果您無法從本地上傳，可以在 CHTC 上直接更新文件。

## 修復 extract_latents.sh

需要在以下位置添加/修改：

### 1. 修復 DEVICE 變量（約第146-160行）

找到這段：
```bash
DEVICE=$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "cpu")
```

替換為：
```bash
DEVICE=$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null)

# Ensure DEVICE has a valid value (default to cuda if empty, as we're requesting GPU)
if [ -z "$DEVICE" ]; then
    echo "Warning: Could not detect device, defaulting to cuda (GPU requested)"
    DEVICE="cuda"
fi
```

### 2. 添加預創建目錄（約第163行之前）

在 `echo "Running extraction..."` 之前添加：
```bash
# Pre-create output directory structure to ensure it exists
OUTPUT_BASE="model_latents"
OUTPUT_DIR="$OUTPUT_BASE/$MODEL_VERSION"
LATENTS_DIR="$OUTPUT_DIR/latents"

echo "Pre-creating output directory structure..."
mkdir -p "$LATENTS_DIR"
echo "✓ Output directory structure created: $OUTPUT_DIR"
```

### 3. 修改輸出目錄變量（約第184行）

將 `--output_dir model_latents` 改為 `--output_dir "$OUTPUT_BASE"`

## 修復 extract_latents_from_home.sub

在 Resources 部分（約第38行之後）添加：
```bash
# Increase job runtime limit to 24 hours (default is 12 hours)
+MaxJobRuntime = 86400
```

## 或者使用 sed 快速修復

在 CHTC 上執行：
```bash
# 備份原文件
cp /staging/groups/bhaskar_group/rho9/extract_latents.sh /staging/groups/bhaskar_group/rho9/extract_latents.sh.backup

# 使用 vi 或 nano 編輯
vi /staging/groups/bhaskar_group/rho9/extract_latents.sh
# 或
nano /staging/groups/bhaskar_group/rho9/extract_latents.sh
```

然後手動應用上述修改。

