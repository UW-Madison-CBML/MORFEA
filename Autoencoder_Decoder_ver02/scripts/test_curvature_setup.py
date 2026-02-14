#!/usr/bin/env python3
"""
Test script to check if all dependencies are available for curvature analysis.
"""

import sys
from pathlib import Path

print("=" * 60)
print("Testing Curvature Analysis Setup")
print("=" * 60)

# Check Python version
print(f"\n1. Python version: {sys.version}")

# Check required packages
required_packages = {
    'numpy': 'numpy',
    'torch': 'torch',
    'matplotlib': 'matplotlib',
    'PIL': 'Pillow',
    'cv2': 'opencv-python'
}

print("\n2. Checking required packages:")
missing = []
for module_name, package_name in required_packages.items():
    try:
        __import__(module_name)
        print(f"   ✓ {package_name}")
    except ImportError:
        print(f"   ✗ {package_name} (missing)")
        missing.append(package_name)

# Check optional packages
print("\n3. Checking optional packages:")
try:
    import tphate
    print("   ✓ tphate")
    tphate_available = True
except ImportError:
    print("   ✗ tphate (not available)")
    tphate_available = False
    try:
        import phate
        print("   ✓ phate (fallback available)")
        phate_available = True
    except ImportError:
        print("   ✗ phate (not available)")
        phate_available = False

# Check if we can import local modules
print("\n4. Checking local modules:")
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dataset_ivf import IVFSequenceDataset
    print("   ✓ dataset_ivf")
except ImportError as e:
    print(f"   ✗ dataset_ivf: {e}")

try:
    from model import ConvLSTMAutoencoder
    print("   ✓ model (ConvLSTMAutoencoder)")
except ImportError as e:
    print(f"   ✗ model: {e}")

try:
    from model import Encoder, Decoder
    print("   ✓ model (Encoder, Decoder)")
except ImportError as e:
    print(f"   ✗ model (Encoder, Decoder): {e}")

# Check for data directory
print("\n5. Checking data directories:")
data_paths = [
    Path('data'),
    Path('/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset'),
    Path('../data'),
]

for path in data_paths:
    if path.exists():
        print(f"   ✓ {path}")
        break
else:
    print("   ✗ No data directory found")
    print("     (This is OK if running on local machine)")

# Check for checkpoint
print("\n6. Checking model checkpoint:")
checkpoint_paths = [
    Path('checkpoints/checkpoint_epoch_50.pt'),
    Path('checkpoint_epoch_50.pt'),
    Path('../checkpoints/checkpoint_epoch_50.pt'),
]

for path in checkpoint_paths:
    if path.exists():
        print(f"   ✓ {path}")
        break
else:
    print("   ✗ No checkpoint found")
    print("     (This is OK if running on local machine)")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    print("   Install with: pip install " + " ".join(missing))
else:
    print("\n✅ All required packages are installed")

if not tphate_available and not phate_available:
    print("\n⚠️  Neither tphate nor phate available")
    print("   Install with: pip install tphate (or phate)")
else:
    print("\n✅ TPHATE/PHATE available")

print("\n💡 Note: This script should be run on CHTC where:")
print("   - Data is available in staging or data/ directory")
print("   - Model checkpoint is available")
print("   - All dependencies are installed")

if missing or (not tphate_available and not phate_available):
    print("\n❌ Setup incomplete - fix issues above before running analysis")
    sys.exit(1)
else:
    print("\n✅ Setup looks good!")

