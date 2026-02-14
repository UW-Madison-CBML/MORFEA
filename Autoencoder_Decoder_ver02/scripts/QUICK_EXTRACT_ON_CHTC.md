# 在 CHTC 上快速解壓 Cells

## 方法 1: 直接執行命令（最快）

在 CHTC 上直接執行以下命令：

```bash
# 1. 設定變數
GROUP_TAR="/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz"
EXTRACT_DIR="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset"
TMPDIR="/staging/groups/bhaskar_group/rho9/tmp"

# 2. 創建目錄
mkdir -p "$EXTRACT_DIR"
mkdir -p "$TMPDIR"
export TMPDIR

# 3. 解壓 ZS435-5
echo "解壓 ZS435-5..."
tar -xzf "$GROUP_TAR" -C "$(dirname $EXTRACT_DIR)" "embryo_dataset/ZS435-5"

# 4. 解壓 RS363-7
echo "解壓 RS363-7..."
tar -xzf "$GROUP_TAR" -C "$(dirname $EXTRACT_DIR)" "embryo_dataset/RS363-7"

# 5. 驗證
echo ""
echo "驗證結果:"
ls -lh "$EXTRACT_DIR"/*/ | head -10
du -sh "$EXTRACT_DIR"
```

## 方法 2: 創建並執行腳本

在 CHTC 上執行：

```bash
cd /staging/groups/bhaskar_group/rho9/ivf_analysis

# 創建腳本目錄
mkdir -p scripts

# 創建腳本（複製以下內容）
cat > scripts/extract_from_group_tar.sh << 'EOF'
#!/bin/bash
set -e

GROUP_TAR_FILE="/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz"
STAGING_DIR="/staging/groups/bhaskar_group/rho9/ivf_data"
EXTRACT_DIR="$STAGING_DIR/embryo_dataset"
export TMPDIR="/staging/groups/bhaskar_group/rho9/tmp"
mkdir -p "$EXTRACT_DIR" "$TMPDIR"

CELLS=("${@:-ZS435-5 RS363-7}")

for cell in "${CELLS[@]}"; do
    echo "解壓: $cell"
    if [ -d "$EXTRACT_DIR/$cell" ]; then
        echo "  已存在，跳過"
        continue
    fi
    tar -xzf "$GROUP_TAR_FILE" -C "$STAGING_DIR" "embryo_dataset/$cell"
    if [ -d "$EXTRACT_DIR/$cell" ]; then
        SIZE=$(du -sh "$EXTRACT_DIR/$cell" | cut -f1)
        COUNT=$(ls "$EXTRACT_DIR/$cell"/*.jpeg "$EXTRACT_DIR/$cell"/*.jpg 2>/dev/null | wc -l)
        echo "  ✓ 完成: $SIZE, $COUNT frames"
    fi
done

echo ""
echo "✅ 完成！資料在: $EXTRACT_DIR"
EOF

# 設定執行權限
chmod +x scripts/extract_from_group_tar.sh

# 執行
bash scripts/extract_from_group_tar.sh
```

## 方法 3: 檢查 tar.gz 內容（確認 cells 存在）

在解壓前，先確認 cells 是否存在：

```bash
# 列出所有 cells
tar -tzf /staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz | \
    grep "^embryo_dataset/" | cut -d'/' -f2 | sort -u | head -20

# 檢查特定 cell
tar -tzf /staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz | \
    grep "^embryo_dataset/ZS435-5/" | head -5
```

## 解壓後執行分析

解壓完成後，執行分析：

```bash
cd /staging/groups/bhaskar_group/rho9/ivf_analysis

# 指定 data_root
python3 scripts/analyze_trajectory_curvature.py \
    --video_name ZS435-5 \
    --data_root /staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset

python3 scripts/analyze_trajectory_curvature.py \
    --video_name RS363-7 \
    --data_root /staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset
```

