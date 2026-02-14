# 在 CHTC 上查找 Aadhitya 的 Latent 文件

## 🔍 直接运行查找命令

在 CHTC 上，直接运行以下命令：

```bash
# 1. 查找 latents.npy
find /staging/groups/bhaskar_group -name "latents.npy" -type f 2>/dev/null

# 2. 查找 latents.csv
find /staging/groups/bhaskar_group -name "latents.csv" -type f 2>/dev/null

# 3. 查找所有包含 "latent" 的文件
find /staging/groups/bhaskar_group -name "*latent*" -type f 2>/dev/null | head -20

# 4. 查找 ivf 相关目录
find /staging/groups/bhaskar_group -type d -name "*ivf*" 2>/dev/null

# 5. 列出可能的 ivf 目录内容
ls -lh /staging/groups/bhaskar_group/ivf/ 2>/dev/null
ls -lh /staging/groups/bhaskar_group/*/ivf/ 2>/dev/null
```

## 📋 检查常见路径

```bash
# 检查这些常见路径
for path in \
  "/staging/groups/bhaskar_group/ivf" \
  "/staging/groups/bhaskar_group/ivf/data" \
  "/staging/groups/bhaskar_group/rho9/ivf" \
  "/staging/groups/bhaskar_group/aadhitya/ivf" \
  "/staging/groups/bhaskar_group/shared/ivf"; do
  if [ -d "$path" ]; then
    echo "=== $path ==="
    ls -lh "$path" 2>/dev/null | head -10
    echo ""
  fi
done
```

## ✅ 验证找到的文件

找到文件后，验证格式：

```bash
# 假设文件在 /staging/groups/bhaskar_group/ivf/
python3 << 'EOF'
import numpy as np
import pandas as pd

npy_path = "/staging/groups/bhaskar_group/ivf/latents.npy"
csv_path = "/staging/groups/bhaskar_group/ivf/latents.csv"

print("=== 验证文件 ===")
Z = np.load(npy_path)
df = pd.read_csv(csv_path)

print(f"NPY shape: {Z.shape}")
print(f"CSV shape: {df.shape}")
print(f"CSV columns: {df.columns.tolist()}")
print(f"Unique embryos: {df['cell_id'].nunique()}")
print(f"Consistent: {len(df) == Z.shape[0]}")
EOF
```






