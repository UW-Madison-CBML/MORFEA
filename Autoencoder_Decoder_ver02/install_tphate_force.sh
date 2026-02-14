#!/bin/bash
# 强制安装 TPHATE（不允许近似）

echo "=== Installing TPHATE (REQUIRED, no approximation) ==="

# 1. 升级基础工具
echo "1. Upgrading numpy, setuptools, wheel..."
pip install --user --upgrade numpy setuptools wheel pip

# 2. 检查 C++ 编译器
echo ""
echo "2. Checking C++ compiler..."
if command -v gcc &> /dev/null; then
    echo "   ✓ gcc found: $(gcc --version | head -1)"
else
    echo "   ⚠️  gcc not found. s_gd2 may fail to compile."
    echo "   On CHTC, you may need to request C++ compiler access."
fi

# 3. 尝试安装 s_gd2
echo ""
echo "3. Installing s_gd2 (required dependency)..."
pip uninstall -y s_gd2 2>/dev/null
pip install --user --no-cache-dir s_gd2

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ s_gd2 installation failed!"
    echo "   This is likely a C++ compilation issue."
    echo ""
    echo "   Solutions:"
    echo "   1. Request C++ compiler access from CHTC support"
    echo "   2. Use a Singularity container with pre-compiled tphate"
    echo "   3. Install on a machine with C++ compiler, then transfer"
    exit 1
fi

# 4. 安装 tphate
echo ""
echo "4. Installing tphate..."
pip install --user --no-cache-dir tphate

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ tphate installation failed!"
    exit 1
fi

# 5. 验证安装
echo ""
echo "5. Verifying installation..."
python3 << 'EOF'
try:
    import tphate
    print("✓ tphate imported successfully")
    try:
        print(f"  Version: {tphate.__version__}")
    except:
        print("  Version: unknown")
    
    # 尝试检查 API
    print("\n  Checking TPHATE API...")
    help(tphate.TPHATE.fit_transform)
except ImportError as e:
    print(f"❌ tphate import failed: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ TPHATE installation complete!"
else
    echo ""
    echo "❌ TPHATE verification failed."
    exit 1
fi

