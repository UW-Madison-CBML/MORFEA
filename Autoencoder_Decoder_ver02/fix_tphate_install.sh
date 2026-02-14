#!/bin/bash
# 修复 TPHATE 安装问题

echo "=== Fixing TPHATE Installation ==="

# 1. 升级 numpy 和 setuptools（解决兼容性问题）
echo "1. Upgrading numpy and setuptools..."
pip install --user --upgrade numpy setuptools wheel

# 2. 尝试重新安装 s_gd2（tphate 的依赖）
echo ""
echo "2. Reinstalling s_gd2..."
pip uninstall -y s_gd2 2>/dev/null
pip install --user --no-cache-dir s_gd2

# 3. 如果 s_gd2 安装失败，尝试安装编译依赖
if [ $? -ne 0 ]; then
    echo ""
    echo "3. s_gd2 installation failed. This may require C++ compiler."
    echo "   On CHTC, you may need to use a different approach."
    echo "   Trying alternative installation..."
fi

# 4. 尝试安装 tphate
echo ""
echo "4. Installing tphate..."
pip install --user --no-cache-dir tphate

# 5. 验证
echo ""
echo "5. Verifying installation..."
python3 << 'EOF'
try:
    import tphate
    print("✓ tphate imported successfully")
    # 尝试检查版本
    try:
        print(f"  Version: {tphate.__version__}")
    except:
        print("  Version: unknown")
except ImportError as e:
    print(f"❌ tphate import failed: {e}")
    print("\nAlternative: You may need to use PHATE + time feature")
    print("  (but TPHATE is strongly recommended for your use case)")
EOF

