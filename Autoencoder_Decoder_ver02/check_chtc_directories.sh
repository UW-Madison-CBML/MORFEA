#!/bin/bash
# check_chtc_directories.sh
# 检查CHTC上的目录结构

echo "=== 检查CHTC目录结构 ==="
echo ""
echo "请在CHTC terminal上运行以下命令："
echo ""
echo "# 1. 检查staging目录"
echo "ls -la /staging/groups/bhaskar_group/rho9/"
echo ""
echo "# 2. 检查home目录"
echo "ls -la ~/ivf_repo/"
echo ""
echo "# 3. 检查checkpoint在哪里"
echo "find /staging/groups/bhaskar_group/rho9 -name 'checkpoint_epoch_50.pt' 2>/dev/null"
echo "find ~/ivf_repo -name 'checkpoint_epoch_50.pt' 2>/dev/null"
echo ""
echo "# 4. 检查index.csv在哪里"
echo "find /staging/groups/bhaskar_group/rho9 -name 'index.csv' 2>/dev/null"
echo "find ~/ivf_repo -name 'index.csv' 2>/dev/null"
echo ""

