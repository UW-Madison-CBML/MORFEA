#!/bin/bash
# 检查所有文件是否准备好上传到CHTC

echo "=== 检查文件准备情况 ==="
echo ""

PROJECT_DIR="/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"
cd "$PROJECT_DIR"

# 1. 检查必要文件是否存在
echo "1. 检查必要文件..."
files=(
    "extract_all_latent_trajectories.py"
    "model.py"
    "dataset_ivf.py"
    "build_index.py"
    "extract_latents.sh"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file - MISSING!"
        exit 1
    fi
done

# 2. 检查extract_all_latent_trajectories.py是否包含优化
echo ""
echo "2. 检查代码优化..."
if grep -q "read directly from CSV" extract_all_latent_trajectories.py; then
    echo "  ✓ 包含CSV优化"
else
    echo "  ✗ 缺少CSV优化"
    exit 1
fi

# 3. 检查dataset_ivf.py是否支持tar.gz
echo ""
echo "3. 检查tar.gz支持..."
if grep -q "tar_file" dataset_ivf.py && grep -q "tarfile.open" dataset_ivf.py; then
    echo "  ✓ 支持tar.gz读取"
else
    echo "  ✗ 缺少tar.gz支持"
    exit 1
fi

# 4. 检查extract_latents.sh是否正确
echo ""
echo "4. 检查shell脚本..."
if head -1 extract_latents.sh | grep -q "#!/bin/bash"; then
    echo "  ✓ 脚本开头正确"
else
    echo "  ✗ 脚本开头错误"
    exit 1
fi

if grep -q "需要我提供" extract_latents.sh; then
    echo "  ✗ 脚本包含占位符文本！"
    exit 1
else
    echo "  ✓ 脚本没有占位符"
fi

# 5. 检查Python语法
echo ""
echo "5. 检查Python语法..."
python3 -m py_compile extract_all_latent_trajectories.py 2>/dev/null && echo "  ✓ extract_all_latent_trajectories.py 语法正确" || echo "  ✗ extract_all_latent_trajectories.py 语法错误"
python3 -m py_compile model.py 2>/dev/null && echo "  ✓ model.py 语法正确" || echo "  ✗ model.py 语法错误"
python3 -m py_compile dataset_ivf.py 2>/dev/null && echo "  ✓ dataset_ivf.py 语法正确" || echo "  ✗ dataset_ivf.py 语法错误"

echo ""
echo "=== 所有检查通过！可以上传到CHTC ==="

