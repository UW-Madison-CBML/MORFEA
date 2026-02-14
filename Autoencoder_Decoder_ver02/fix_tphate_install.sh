#!/bin/bash

echo "=== Fixing TPHATE Installation ==="

echo "1. Upgrading numpy and setuptools..."
pip install --user --upgrade numpy setuptools wheel

echo ""
echo "2. Reinstalling s_gd2..."
pip uninstall -y s_gd2 2>/dev/null
pip install --user --no-cache-dir s_gd2

if [ $? -ne 0 ]; then
    echo ""
    echo "3. s_gd2 installation failed. This may require C++ compiler."
    echo "   On CHTC, you may need to use a different approach."
    echo "   Trying alternative installation..."
fi

echo ""
echo "4. Installing tphate..."
pip install --user --no-cache-dir tphate

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
except ImportError as e:
    print(f"❌ tphate import failed: {e}")
    print("\nAlternative: You may need to use PHATE + time feature")
    print("  (but TPHATE is strongly recommended for your use case)")
EOF

