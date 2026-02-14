#!/bin/bash
set -euo pipefail

echo "===== RUN_TRAIN START ====="
echo "Hostname: $(hostname)"
echo "CWD: $(pwd)"
echo "Initial files:"
ls

########################################################
# Step 0: 確保這個 Python 裡有必要的套件
########################################################
echo
echo "Step 0: 安裝 / 確認必要的套件是否可用（在執行節點上）..."

python - << 'PYEOF'
import importlib
import subprocess
import sys
import site
import os

# 確保用戶 site-packages 在 sys.path 中
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)
    print(f"[dep] 添加用戶 site-packages 到 Python 路徑: {user_site}")

# 也添加 ~/.local/lib/python*/site-packages
home = os.path.expanduser("~")
local_lib = os.path.join(home, ".local", "lib")
if os.path.exists(local_lib):
    for version_dir in os.listdir(local_lib):
        if version_dir.startswith("python"):
            site_packages = os.path.join(local_lib, version_dir, "site-packages")
            if os.path.exists(site_packages) and site_packages not in sys.path:
                sys.path.insert(0, site_packages)
                print(f"[dep] 添加本地 site-packages: {site_packages}")

def ensure(pkg: str):
    # 嘗試導入
    try:
        mod = importlib.import_module(pkg)
        if hasattr(mod, "__version__"):
            print(f"[dep] {pkg} already安裝好了 (版本: {mod.__version__})，可以直接用")
        else:
            print(f"[dep] {pkg} already安裝好了，可以直接用")
        return True
    except ImportError:
        print(f"[dep] {pkg} 不存在，開始用 pip --user 安裝...")

    # 安裝包
    try:
        subprocess.check_call([
            sys.executable,
            "-m", "pip", "install", "--user", "--no-cache-dir", pkg
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[dep] {pkg} pip 安裝完成")
    except subprocess.CalledProcessError as e:
        print(f"[dep] {pkg} pip 安裝失敗: {e}")
        return False

    # 重新加載用戶 site-packages
    importlib.invalidate_caches()
    
    # 重新添加用戶 site-packages（可能剛創建）
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.insert(0, user_site)
    
    # 驗證導入
    try:
        mod = importlib.import_module(pkg)
        if hasattr(mod, "__version__"):
            print(f"[dep] {pkg} 安裝成功，可以 import 囉 (版本: {mod.__version__})")
        else:
            print(f"[dep] {pkg} 安裝成功，可以 import 囉")
        return True
    except ImportError:
        print(f"[dep] {pkg} 安裝後仍無法導入，可能是路徑問題")
        print(f"[dep] sys.path = {sys.path}")
        return False

# 安裝必要的套件
print("[dep] 開始檢查並安裝必要的套件...")
for name in ["pandas", "torch", "torchvision", "numpy", "tqdm"]:
    if not ensure(name):
        print(f"[dep] ERROR: {name} 安裝失敗，繼續嘗試其他套件...")

# Pillow 需要特殊處理（導入時使用 PIL）
print("[dep] 檢查 Pillow (導入時使用 PIL)...")
try:
    importlib.import_module("PIL")
    print("[dep] PIL already安裝好了，可以直接用")
except ImportError:
    print("[dep] PIL 不存在，安裝 Pillow...")
    try:
        subprocess.check_call([
            sys.executable,
            "-m", "pip", "install", "--user", "--no-cache-dir", "Pillow"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("[dep] Pillow pip 安裝完成")
        importlib.invalidate_caches()
        user_site = site.getusersitepackages()
        if user_site and user_site not in sys.path:
            sys.path.insert(0, user_site)
        try:
            importlib.import_module("PIL")
            print("[dep] Pillow 安裝成功，可以 import PIL 囉")
        except ImportError:
            print("[dep] Pillow 安裝後仍無法導入 PIL")
            print(f"[dep] sys.path = {sys.path}")
    except subprocess.CalledProcessError as e:
        print(f"[dep] Pillow pip 安裝失敗: {e}")

# opencv 不需要（代碼中沒有使用）
print("[dep] 跳過 opencv（代碼中未使用）")

print("[dep] 套件檢查完成")
PYEOF

########################################################
# Step 1–5: staging ➜ 解壓 ➜ 建 index ➜ 訓練
########################################################

STAGING_TAR="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"
WORK_TAR="$PWD/embryo_dataset.tar.gz"
RAW_DIR="$PWD/data_raw"
DATA_LINK="$PWD/data"
INDEX_CSV="$PWD/index.csv"

echo
echo "Step 1: 檢查 staging 掛載與 tar 檔存在..."
mount | grep /staging || echo "[WARN] /staging 沒有在 mount 列表（但目前看起來是有的）"
if [ ! -f "$STAGING_TAR" ]; then
    echo "✗ ERROR: staging tar 不存在: $STAGING_TAR"
    # 嘗試備用路徑
    STAGING_TAR="/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz"
    if [ ! -f "$STAGING_TAR" ]; then
        echo "✗ ERROR: 備用路徑也不存在: $STAGING_TAR"
        exit 1
    fi
fi
ls -lh "$STAGING_TAR"

echo
echo "Step 2: 從 staging 複製 tar 到工作目錄..."
rm -f "$WORK_TAR"
rm -rf "$RAW_DIR" "$DATA_LINK"
cp "$STAGING_TAR" "$WORK_TAR"
echo "  工作目錄中的檔案大小："
ls -lh "$WORK_TAR"

echo
echo "Step 3: 解壓 embryo_dataset.tar.gz 到 $RAW_DIR ..."
mkdir -p "$RAW_DIR"

set +e
tar --warning=no-unknown-keyword -xzf "$WORK_TAR" -C "$RAW_DIR"
TAR_STATUS=$?
set -e

if [ "$TAR_STATUS" -ne 0 ]; then
  echo "[WARN] tar -xzf 失敗，改用 tar -xvf（假設不是 gzip 壓縮）..."
  tar --warning=no-unknown-keyword -xvf "$WORK_TAR" -C "$RAW_DIR"
fi

echo "  解壓後 RAW_DIR 內容："
ls -lh "$RAW_DIR"
ls -lh "$RAW_DIR/embryo_dataset" || true

echo
echo "Step 3.5: 建立 data symlink（讓 dataset_ivf 可以找到影像資料）..."
if [ -d "$RAW_DIR/embryo_dataset" ]; then
  TARGET_DIR="$RAW_DIR/embryo_dataset"
else
  TARGET_DIR="$RAW_DIR"
fi

ln -s "$TARGET_DIR" "$DATA_LINK"
echo "  data -> $TARGET_DIR"
ls -ld "$DATA_LINK"

echo
echo "Step 4: 產生 index.csv ..."
rm -f "$INDEX_CSV"

python -u build_index.py --root "$DATA_LINK" --out "$INDEX_CSV"

if [ ! -f "$INDEX_CSV" ]; then
  echo "[ERROR] build_index.py 跑完還是沒有 index.csv，退出。"
  exit 1
fi

echo "  產生的 index.csv："
ls -lh "$INDEX_CSV"

echo
echo "Step 5: 開始訓練..."
python -u train.py

echo "===== RUN_TRAIN DONE ====="
