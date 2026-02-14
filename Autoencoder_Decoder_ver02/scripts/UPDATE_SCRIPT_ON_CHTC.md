# 在 CHTC 上更新腳本

## 方法 1: 檢查並添加必要的 import

在 CHTC 上執行：

```bash
cd /staging/groups/bhaskar_group/rho9/ivf_analysis/scripts

# 檢查是否已有 tarfile 和 io
grep -n "import tarfile\|import io" analyze_trajectory_curvature.py

# 如果沒有，在 import 區塊添加
# 找到 "from PIL import Image" 這一行，在後面添加：
```

如果沒有找到 `import tarfile` 和 `import io`，需要手動添加。找到這一行：
```python
from PIL import Image
```

在它後面添加：
```python
import tarfile
import io
```

## 方法 2: 使用 patch 文件

在 CHTC 上創建一個 patch 文件來更新腳本：

```bash
cd /staging/groups/bhaskar_group/rho9/ivf_analysis/scripts

# 檢查是否需要添加 import
if ! grep -q "import tarfile" analyze_trajectory_curvature.py; then
    # 在 "from PIL import Image" 後面添加
    sed -i '/from PIL import Image/a import tarfile\nimport io' analyze_trajectory_curvature.py
    echo "✓ 已添加 tarfile 和 io import"
else
    echo "✓ import 已存在"
fi
```

## 方法 3: 直接測試（如果腳本已更新）

如果腳本已經有 `_load_from_tar` 函數，可以直接測試：

```bash
cd /staging/groups/bhaskar_group/rho9/ivf_analysis

# 測試腳本
python3 scripts/analyze_trajectory_curvature.py --video_name ZS435-5
```

如果出現 `NameError: name 'tarfile' is not defined`，則需要添加 import。

## 快速檢查

執行這個命令來檢查腳本狀態：

```bash
cd /staging/groups/bhaskar_group/rho9/ivf_analysis/scripts

echo "檢查 import:"
grep -E "import tarfile|import io" analyze_trajectory_curvature.py || echo "❌ 缺少 import"

echo ""
echo "檢查 _load_from_tar 函數:"
grep -n "def _load_from_tar" analyze_trajectory_curvature.py || echo "❌ 缺少函數"

echo ""
echo "檢查 load_latent_vectors_for_video 是否調用 _load_from_tar:"
grep -n "_load_from_tar" analyze_trajectory_curvature.py | head -3
```

