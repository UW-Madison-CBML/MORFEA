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

print("=== 检查依赖包 ===")
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
    print(f"缺少的包: {', '.join(missing)}")
    print()
    print("在CHTC上，PyTorch容器通常包含:")
    print("  ✓ torch (包含在容器中)")
    print("  ✓ numpy (包含在容器中)")
    print("  ✗ pandas (可能需要安装)")
    print("  ✗ Pillow (可能需要安装)")
    print()
    print("解决方案:")
    print("1. 使用包含所有包的容器")
    print("2. 或者在脚本中安装缺失的包")
else:
    print("✓ 所有依赖包都已安装")

