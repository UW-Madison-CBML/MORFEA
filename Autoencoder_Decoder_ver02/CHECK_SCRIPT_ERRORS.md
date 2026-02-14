# 检查脚本错误

## 问题
进程退出（Exit 1）但没有生成任何输出。

## 检查方法

### 方法1: 直接运行查看错误

```bash
cd /staging/groups/bhaskar_group/rho9

# 直接运行，查看错误信息
python3 generate_tphate_for_aadhitya.py \
    --npy_file /staging/groups/bhaskar_group/ivf/latents/latents.npy \
    --csv_file /staging/groups/bhaskar_group/ivf/latents/latents.csv \
    --output_base /staging/groups/bhaskar_group/ivf/v1_baseline_tphate \
    --knn 5
```

### 方法2: 检查脚本是否存在和可执行

```bash
# 检查脚本
ls -l /staging/groups/bhaskar_group/rho9/generate_tphate_for_aadhitya.py

# 检查 Python 模块
python3 -c "import tphate; print('TPHATE OK')"
python3 -c "import numpy; print('NumPy OK')"
python3 -c "import pandas; print('Pandas OK')"
python3 -c "import matplotlib; print('Matplotlib OK')"
```

### 方法3: 测试脚本语法

```bash
python3 -m py_compile /staging/groups/bhaskar_group/rho9/generate_tphate_for_aadhitya.py
```

## 可能的问题

1. **缺少依赖包**（如 tphate）
2. **脚本语法错误**
3. **输入文件路径错误**
4. **权限问题**

先运行方法1，查看具体的错误信息。






