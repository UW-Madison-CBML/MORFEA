#!/bin/bash

echo "=== Installing TPHATE ==="

echo "1. Upgrading numpy..."
pip install --user --upgrade numpy

echo ""
echo "2. Installing tphate..."
pip install --user tphate

if [ $? -ne 0 ]; then
    echo ""
    echo "3. Standard install failed, trying from source..."
    pip install --user git+https://github.com/KrishnaswamyLab/tphate.git
fi

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

