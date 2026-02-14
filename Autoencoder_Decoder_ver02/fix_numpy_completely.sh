#!/bin/bash

cd ~/ivf_repo

echo "=== Step 1: Check current NumPy ==="
python3 -c "import numpy as np; print(f'Current NumPy: {np.__version__}')" 2>&1

echo ""
echo "=== Step 2: Force uninstall NumPy 2.x ==="
pip uninstall --user -y numpy 2>/dev/null || true
pip uninstall --user -y s_gd2 tphate 2>/dev/null || true

echo ""
echo "=== Step 3: Install NumPy 1.24.3 (compatible version) ==="
pip install --user "numpy==1.24.3" --no-cache-dir --force-reinstall

echo ""
echo "=== Step 4: Verify NumPy ==="
python3 -c "import numpy as np; print(f'New NumPy: {np.__version__}')" || {
    echo "❌ NumPy import failed!"
    exit 1
}

echo ""
echo "=== Step 5: Reinstall s_gd2 (will recompile for NumPy 1.24.3) ==="
pip install --user s_gd2 --no-cache-dir --force-reinstall --no-deps || {
    echo "⚠️  s_gd2 install failed, trying with dependencies..."
    pip install --user s_gd2 --no-cache-dir --force-reinstall
}

echo ""
echo "=== Step 6: Reinstall tphate ==="
pip install --user tphate --no-cache-dir --force-reinstall --no-deps || {
    echo "⚠️  tphate install failed, trying with dependencies..."
    pip install --user tphate --no-cache-dir --force-reinstall
}

echo ""
echo "=== Step 7: Test tphate import ==="
python3 << 'PYTHON'
import sys
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} imported")
    
    import tphate
    print("✓ tphate imported successfully!")
    
    # Test creating a TPHATE object
    tph = tphate.TPHATE(n_components=3, knn=10)
    print("✓ TPHATE object created successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ NumPy and tphate are fixed!"
    echo "=========================================="
    echo ""
    echo "Now run TPHATE pipeline:"
    echo "  python3 tphate_3d_pipeline.py \\"
    echo "      --input latents_preprocessed_direct.npz \\"
    echo "      --output tphate_3d_results_direct.npz \\"
    echo "      --use_pca \\"
    echo "      --knn 10 \\"
    echo "      --n_components 3"
else
    echo ""
    echo "❌ Fix failed! Try manual steps:"
    echo "  1. pip uninstall --user -y numpy s_gd2 tphate"
    echo "  2. pip install --user numpy==1.24.3"
    echo "  3. pip install --user s_gd2 --force-reinstall"
    echo "  4. pip install --user tphate --force-reinstall"
    exit 1
fi

