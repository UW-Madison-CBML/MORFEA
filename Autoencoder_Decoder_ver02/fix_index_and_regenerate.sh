#!/bin/bash
# Fix index.csv paths and regenerate reconstructions on CHTC

echo "=== 修复 index.csv 并重新生成重建示例 ==="
echo ""

cd ~/ivf_repo

# 1. 检查 data symlink
echo "1. 检查 data symlink..."
if [ -L "data" ]; then
    echo "   ✓ data symlink 存在"
    ls -ld data
else
    echo "   ⚠️  data symlink 不存在，创建中..."
    # 检查 staging 数据是否存在
    if [ -d "/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset" ]; then
        ln -sf /staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset data
        echo "   ✓ 已创建 data -> /staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset"
    elif [ -d "/staging/groups/bhaskar_group/ivf/embryo_dataset" ]; then
        ln -sf /staging/groups/bhaskar_group/ivf/embryo_dataset data
        echo "   ✓ 已创建 data -> /staging/groups/bhaskar_group/ivf/embryo_dataset"
    else
        echo "   ❌ 找不到数据目录，请检查 staging 路径"
        exit 1
    fi
fi

# 2. 备份旧的 index.csv
if [ -f "index.csv" ]; then
    echo ""
    echo "2. 备份旧的 index.csv..."
    cp index.csv index.csv.backup_$(date +%Y%m%d_%H%M%S)
    echo "   ✓ 已备份"
fi

# 3. 重新生成 index.csv
echo ""
echo "3. 重新生成 index.csv（使用当前可访问的数据路径）..."
python3 build_index.py --root data --out index.csv

if [ ! -f "index.csv" ]; then
    echo "   ❌ index.csv 生成失败"
    exit 1
fi

echo "   ✓ index.csv 已重新生成"
echo "   行数: $(wc -l < index.csv)"

# 4. 重新生成重建示例
echo ""
echo "4. 重新生成重建示例..."
rm -rf reconstructions/*.png 2>/dev/null

python3 generate_reconstructions.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --index_csv index.csv \
    --output_dir reconstructions \
    --num_samples 5 \
    --n_frames 10

echo ""
echo "=== 完成！ ==="
echo "重建示例保存在: reconstructions/"

