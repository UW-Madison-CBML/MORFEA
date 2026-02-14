#!/usr/bin/env python3
"""
检查脚本需要的所有依赖包
"""
import sys

required_packages = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'PIL': 'Pillow (PIL)',
}

print()

missing = []
for package, name in required_packages.items():
    try:
        if package == 'PIL':
            import PIL
            print(f"✓ {name} (PIL)")
        else:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
    except ImportError:
        print(f"✗ {name} - MISSING")
        missing.append(name)

print()
if missing:
    print()
    print()
else:

