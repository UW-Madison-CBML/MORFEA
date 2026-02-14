#!/bin/bash

echo ""

condor_q
echo ""

echo "   extract_latents.sh:"
if [ -f /staging/groups/bhaskar_group/rho9/extract_latents.sh ]; then
    head -3 /staging/groups/bhaskar_group/rho9/extract_latents.sh
    if grep -q "需要我提供" /staging/groups/bhaskar_group/rho9/extract_latents.sh; then
    else
    fi
else
fi
echo ""

for file in extract_all_latent_trajectories.py model.py dataset_ivf.py; do
    if [ -f /staging/groups/bhaskar_group/rho9/$file ]; then
    else
    fi
done
echo ""

if grep -q "read directly from CSV" /staging/groups/bhaskar_group/rho9/extract_all_latent_trajectories.py; then
else
fi
echo ""

if [ -f /staging/groups/bhaskar_group/rho9/index.csv ]; then
    if grep -q "cell_id" /staging/groups/bhaskar_group/rho9/index.csv; then
    else
    fi
else
fi
echo ""

if [ -f /staging/groups/bhaskar_group/rho9/checkpoints/checkpoint_epoch_50.pt ]; then
    ls -lh /staging/groups/bhaskar_group/rho9/checkpoints/checkpoint_epoch_50.pt
else
fi
echo ""

for log in logs/extract_latents_v1_baseline.out logs/extract_latents_v1_baseline.err logs/extract_latents_v1_baseline.log; do
    if [ -f ~/$log ]; then
        echo "   $log:"
    else
    fi
done
echo ""

echo ""

echo ""

echo ""

RESULT_DIR="/staging/groups/bhaskar_group/rho9/model_latents/v1_baseline"
if [ -d "$RESULT_DIR" ]; then
    ls -lh "$RESULT_DIR"
    if [ -d "$RESULT_DIR/latents" ]; then
        COUNT=$(ls -1 "$RESULT_DIR/latents"/*.npy 2>/dev/null | wc -l)
    fi
else
fi
echo ""

python3 << 'PYEOF'
import sys
sys.path.insert(0, '/staging/groups/bhaskar_group/rho9')
try:
    import extract_all_latent_trajectories
    print("   ✓ extract_all_latent_trajectories 可以导入")
except Exception as e:
    print(f"   ❌ 导入失败: {e}")

try:
    import model
    print("   ✓ model 可以导入")
except Exception as e:
    print(f"   ❌ 导入失败: {e}")

try:
    import dataset_ivf
    print("   ✓ dataset_ivf 可以导入")
except Exception as e:
    print(f"   ❌ 导入失败: {e}")
PYEOF

echo ""

