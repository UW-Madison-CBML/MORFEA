#!/bin/bash
# 在 CHTC 上生成重建示例（使用正确的模型文件）

echo "=== 在 CHTC 上生成重建示例 ==="
echo ""

cd ~/ivf_repo

# 检查模型文件
if [ ! -f "model.py" ]; then
    echo "❌ model.py 不存在"
    exit 1
fi

# 检查 checkpoint
if [ ! -f "checkpoints/checkpoint_epoch_50.pt" ]; then
    echo "❌ checkpoint 不存在"
    exit 1
fi

# 检查数据
if [ ! -L "data" ] && [ ! -d "data" ]; then
    echo "创建 data symlink..."
    if [ -d "/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset" ]; then
        ln -sf /staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset data
    elif [ -d "/staging/groups/bhaskar_group/ivf/embryo_dataset" ]; then
        ln -sf /staging/groups/bhaskar_group/ivf/embryo_dataset data
    else
        echo "❌ 找不到数据目录"
        exit 1
    fi
fi

# 检查 index.csv
if [ ! -f "index.csv" ] || [ $(head -1 index.csv | grep -c "/var/lib/condor") -gt 0 ]; then
    echo "重新生成 index.csv..."
    python3 build_index.py --root data --out index.csv
fi

# 生成重建示例
echo ""
echo "开始生成重建示例..."
python3 generate_reconstructions.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --index_csv index.csv \
    --output_dir reconstructions \
    --num_samples 5 \
    --n_frames 10

echo ""
echo "=== 完成！ ==="
echo "重建示例保存在: reconstructions/"

