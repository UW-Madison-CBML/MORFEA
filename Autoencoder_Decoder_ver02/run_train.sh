#!/bin/bash
set -euo pipefail

echo "===== RUN_TRAIN START ====="
echo "Hostname: $(hostname)"
echo "CWD: $(pwd)"
echo "Initial files:"
ls

########################################################
########################################################
echo

python - << 'PYEOF'
import importlib
import subprocess
import sys
import site
import os

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)
    print(f"[dep] 添加用戶 site-packages 到 Python 路徑: {user_site}")

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
    try:
        mod = importlib.import_module(pkg)
        if hasattr(mod, "__version__"):
            print(f"[dep] {pkg} already安裝好了 (版本: {mod.__version__})，可以直接用")
        else:
            print(f"[dep] {pkg} already安裝好了，可以直接用")
        return True
    except ImportError:
        print(f"[dep] {pkg} 不存在，開始用 pip --user 安裝...")

    try:
        subprocess.check_call([
            sys.executable,
            "-m", "pip", "install", "--user", "--no-cache-dir", pkg
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[dep] {pkg} pip 安裝完成")
    except subprocess.CalledProcessError as e:
        print(f"[dep] {pkg} pip 安裝失敗: {e}")
        return False

    importlib.invalidate_caches()
    
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.insert(0, user_site)
    
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

print("[dep] 開始檢查並安裝必要的套件...")
for name in ["pandas", "torch", "torchvision", "numpy", "tqdm"]:
    if not ensure(name):
        print(f"[dep] ERROR: {name} 安裝失敗，繼續嘗試其他套件...")

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

print("[dep] 跳過 opencv（代碼中未使用）")

print("[dep] 套件檢查完成")
PYEOF

########################################################
########################################################

STAGING_TAR="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"
WORK_TAR="$PWD/embryo_dataset.tar.gz"
RAW_DIR="$PWD/data_raw"
DATA_LINK="$PWD/data"
INDEX_CSV="$PWD/index.csv"

echo
if [ ! -f "$STAGING_TAR" ]; then
    STAGING_TAR="/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz"
    if [ ! -f "$STAGING_TAR" ]; then
        exit 1
    fi
fi
ls -lh "$STAGING_TAR"

echo
rm -f "$WORK_TAR"
rm -rf "$RAW_DIR" "$DATA_LINK"
cp "$STAGING_TAR" "$WORK_TAR"
ls -lh "$WORK_TAR"

echo
mkdir -p "$RAW_DIR"

set +e
tar --warning=no-unknown-keyword -xzf "$WORK_TAR" -C "$RAW_DIR"
TAR_STATUS=$?
set -e

if [ "$TAR_STATUS" -ne 0 ]; then
  tar --warning=no-unknown-keyword -xvf "$WORK_TAR" -C "$RAW_DIR"
fi

ls -lh "$RAW_DIR"
ls -lh "$RAW_DIR/embryo_dataset" || true

echo
if [ -d "$RAW_DIR/embryo_dataset" ]; then
  TARGET_DIR="$RAW_DIR/embryo_dataset"
else
  TARGET_DIR="$RAW_DIR"
fi

ln -s "$TARGET_DIR" "$DATA_LINK"
echo "  data -> $TARGET_DIR"
ls -ld "$DATA_LINK"

echo
rm -f "$INDEX_CSV"

python -u build_index.py --root "$DATA_LINK" --out "$INDEX_CSV"

if [ ! -f "$INDEX_CSV" ]; then
  exit 1
fi

ls -lh "$INDEX_CSV"

echo
python -u train.py

echo "===== RUN_TRAIN DONE ====="
