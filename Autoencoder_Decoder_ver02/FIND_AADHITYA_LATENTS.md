# 找到 Aadhitya 的 Latent 文件

## 📍 文件位置

根据 Aadhitya 的消息，文件在：
- **Group storage 的 ivf 目录**
- `latents.npy` - Latent vectors
- `latents.csv` - 包含 `cell_id` 和 `time_step` 列

## 🔍 可能的路径

### 在 CHTC 上：

```bash
# 检查 group storage 的常见位置
/staging/groups/bhaskar_group/ivf/latents.npy
/staging/groups/bhaskar_group/ivf/latents.csv

# 或者可能在子目录
/staging/groups/bhaskar_group/ivf/data/latents.npy
/staging/groups/bhaskar_group/ivf/latents/latents.npy
```

### 检查命令：

```bash
# SSH 到 CHTC
ssh rho9@ap2001.chtc.wisc.edu

# 查找文件
find /staging/groups/bhaskar_group -name "latents.npy" 2>/dev/null
find /staging/groups/bhaskar_group -name "latents.csv" 2>/dev/null

# 或者列出 ivf 目录
ls -lh /staging/groups/bhaskar_group/ivf/
ls -lh /staging/groups/bhaskar_group/*/ivf/ 2>/dev/null
```

## 📋 文件格式

根据 Aadhitya 的描述：
- `latents.npy`: shape 应该是 `[N, latent_dim]`，其中 N 是总帧数
- `latents.csv`: 包含两列
  - `cell_id`: 每个 frame 对应的胚胎 ID
  - `time_step`: 时间步（可选，可能用于排序）

## ✅ 验证文件

```python
import numpy as np
import pandas as pd

# 加载文件
Z = np.load("/staging/groups/bhaskar_group/ivf/latents.npy")
df = pd.read_csv("/staging/groups/bhaskar_group/ivf/latents.csv")

print(f"Latent array shape: {Z.shape}")
print(f"CSV shape: {df.shape}")
print(f"CSV columns: {df.columns.tolist()}")
print(f"Unique embryos: {df['cell_id'].nunique()}")

# 验证一致性
assert len(df) == Z.shape[0], f"Mismatch: {len(df)} rows in CSV but {Z.shape[0]} vectors in .npy"
```

## 🚀 直接使用

如果文件格式已经符合 `export_signatures.py` 的要求，你可以：

1. **检查格式是否匹配：**
   ```python
   # export_signatures.py 期望：
   # - latents/{model_name}.npy
   # - latents/{model_name}.csv (只需要 cell_id 列)
   ```

2. **如果需要转换：**
   - 如果 `latents.csv` 有额外的 `time_step` 列，只需要确保有 `cell_id` 列即可
   - 可能需要重命名或复制文件到 `latents/` 目录

3. **运行 Path Signature 计算：**
   ```bash
   # 假设文件已经准备好
   python export_signatures.py --name aadhitya_v1
   ```

## 🔄 如果需要转换格式

如果文件格式不完全匹配，可以使用转换脚本：

```python
import numpy as np
import pandas as pd
from pathlib import Path

# 读取 Aadhitya 的文件
Z = np.load("/staging/groups/bhaskar_group/ivf/latents.npy")
df = pd.read_csv("/staging/groups/bhaskar_group/ivf/latents.csv")

# 确保有 cell_id 列
if 'cell_id' not in df.columns:
    raise ValueError("CSV must have 'cell_id' column")

# 只保留 cell_id 列（如果需要）
df_output = df[['cell_id']].copy()

# 保存到 latents/ 目录
output_dir = Path("latents")
output_dir.mkdir(exist_ok=True)

np.save(output_dir / "aadhitya_v1.npy", Z)
df_output.to_csv(output_dir / "aadhitya_v1.csv", index=False)

print(f"✓ Converted files saved to latents/")
print(f"  - aadhitya_v1.npy: {Z.shape}")
print(f"  - aadhitya_v1.csv: {df_output.shape}")
```

## 📝 下一步

1. **找到文件位置** - 使用上面的 find 命令
2. **验证文件格式** - 运行验证脚本
3. **如果需要，转换格式** - 使用上面的转换代码
4. **运行 Path Signature** - 使用 `export_signatures.py`






