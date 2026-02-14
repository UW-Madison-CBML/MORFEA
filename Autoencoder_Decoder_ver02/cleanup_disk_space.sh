#!/bin/bash
# 清理磁盘空间并检查大文件

cd ~/ivf_repo

echo "=== Disk Space Check ==="
df -h ~ | tail -1
echo ""

echo "=== Finding Large Files in ~/ivf_repo ==="
du -h --max-depth=2 ~/ivf_repo | sort -hr | head -20
echo ""

echo "=== Checking for Old Checkpoints ==="
if [ -d "checkpoints" ]; then
    echo "Checkpoints directory:"
    ls -lh checkpoints/ | head -10
    echo ""
    echo "Total checkpoint size:"
    du -sh checkpoints/
    echo ""
    echo "⚠️  You can delete old checkpoints (keep only epoch 50):"
    echo "  rm -f checkpoints/checkpoint_epoch_{1..49}.pt"
    echo "  rm -f checkpoints/checkpoint_epoch_{51..100}.pt 2>/dev/null || true"
fi

echo ""
echo "=== Checking for Old TPHATE Files ==="
ls -lh tphate_*.npz latents_*.npz 2>/dev/null | head -10
echo ""

echo "=== Checking Python Cache ==="
find ~/ivf_repo -type d -name "__pycache__" -exec du -sh {} \; 2>/dev/null | head -10
echo ""

echo "=== Checking .pyc Files ==="
find ~/ivf_repo -name "*.pyc" -exec du -sh {} \; 2>/dev/null | head -10
echo ""

echo "=== Recommendations ==="
echo "1. Delete old checkpoints (keep only epoch 50):"
echo "   rm -f checkpoints/checkpoint_epoch_{1..49}.pt"
echo ""
echo "2. Delete Python cache:"
echo "   find ~/ivf_repo -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null"
echo "   find ~/ivf_repo -name '*.pyc' -delete 2>/dev/null"
echo ""
echo "3. Move large files to /staging/ or /project/ if available:"
echo "   mkdir -p /staging/groups/bhaskar_group/rho9/ivf_results"
echo "   mv tphate_segments_direct /staging/groups/bhaskar_group/rho9/ivf_results/ 2>/dev/null || true"
echo ""
echo "4. Or run TPHATE output directly to staging:"
echo "   (modify scripts to output to /staging/...)"

