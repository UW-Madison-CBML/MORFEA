#!/bin/bash
# 修复 NumPy 版本兼容性问题并重新安装 tphate

echo "=== Fixing NumPy and TPHATE Installation ==="
echo ""

cd ~/ivf_repo

# 1. 降级 NumPy 到 1.x
echo "1. Downgrading NumPy to < 2.0..."
pip install --user "numpy<2.0" --force-reinstall --no-cache-dir

if [ $? -ne 0 ]; then
    echo "❌ NumPy downgrade failed!"
    exit 1
fi

echo "✓ NumPy downgraded"
echo ""

# 2. 卸载并重新安装 s_gd2 和 tphate
echo "2. Reinstalling s_gd2 and tphate..."
pip uninstall --user -y s_gd2 tphate 2>/dev/null || true

# 先安装 s_gd2（可能需要从源码编译）
echo "  Installing s_gd2..."
pip install --user s_gd2 --no-cache-dir --force-reinstall

# 然后安装 tphate
echo "  Installing tphate..."
pip install --user tphate --no-cache-dir --force-reinstall

if [ $? -ne 0 ]; then
    echo "  ⚠️  Standard install failed, trying from source..."
    pip install --user git+https://github.com/KrishnaswamyLab/tphate.git --no-cache-dir
fi

echo "✓ tphate installed"
echo ""

# 3. 验证安装
echo "3. Verifying installation..."
python3 -c "import numpy as np; print(f'NumPy version: {np.__version__}')"
python3 -c "import tphate; print('✓ tphate imported successfully')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Installation Complete!"
    echo "=========================================="
    echo ""
    echo "Now run: bash rerun_tphate_pipeline.sh"
else
    echo ""
    echo "❌ Installation verification failed!"
    echo "Try manual installation:"
    echo "  pip install --user 'numpy<2.0' --force-reinstall"
    echo "  pip install --user tphate --force-reinstall"
    exit 1
fi

