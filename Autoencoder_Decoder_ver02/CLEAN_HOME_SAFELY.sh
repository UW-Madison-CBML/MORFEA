#!/bin/bash
# 安全清理 home 目录

echo "=== 可以安全删除的内容 ==="
echo ""
echo "1. .cache 目录 (124M) - 缓存文件，可以删除"
du -sh ~/.cache

echo ""
echo "2. ivf-embryo-analysis-Raffael.tgz (555M) - 压缩文件，如果不需要可以删除"
du -sh ~/ivf-embryo-analysis-Raffael.tgz

echo ""
echo "3. 检查是否有重复的虚拟环境"
echo "   .venv: $(du -sh ~/.venv 2>/dev/null | awk '{print $1}')"
echo "   Desktop/.venv: $(du -sh ~/Desktop/.venv 2>/dev/null | awk '{print $1}')"
echo "   .local: $(du -sh ~/.local 2>/dev/null | awk '{print $1}')"

echo ""
echo "=== 删除建议 ==="
echo ""
echo "# 1. 删除 .cache（安全）"
echo "rm -rf ~/.cache"
echo ""
echo "# 2. 删除压缩文件（如果不需要）"
echo "rm ~/ivf-embryo-analysis-Raffael.tgz"
echo ""
echo "# 3. 如果 Desktop/.venv 和 .venv 重复，可以删除一个"
echo "# rm -rf ~/Desktop/.venv  # 或 ~/.venv"
echo ""
echo "删除后检查配额："
echo "quota -s"






