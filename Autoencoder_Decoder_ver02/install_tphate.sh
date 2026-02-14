#!/bin/bash
# 安装 TPHATE 的脚本（处理依赖问题）

echo "=== Installing TPHATE ==="

# 1. 升级 numpy（解决 s_gd2 的兼容性问题）
echo "1. Upgrading numpy..."
pip install --user --upgrade numpy

# 2. 尝试安装 tphate
echo ""
echo "2. Installing tphate..."
pip install --user tphate

# 3. 如果失败，尝试从源码安装
if [ $? -ne 0 ]; then
    echo ""
    echo "3. Standard install failed, trying from source..."
    pip install --user git+https://github.com/KrishnaswamyLab/tphate.git
fi

# 4. 验证安装
echo ""
echo "4. Verifying installation..."
python3 -c "import tphate; print('✓ tphate installed successfully')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ TPHATE installation complete!"
else
    echo ""
    echo "❌ TPHATE installation failed."
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check if you have C++ compiler: gcc --version"
    echo "  2. Try: pip install --user --upgrade setuptools wheel"
    echo "  3. Check tphate GitHub: https://github.com/KrishnaswamyLab/tphate"
fi

