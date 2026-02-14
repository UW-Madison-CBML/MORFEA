#!/bin/bash
# 查找 Jens 的联系方式

echo "============================================================"
echo "查找 Jens (jlundsgaard) 的联系方式"
echo "============================================================"
echo ""

echo "1. 查找用户信息（如果有finger服务）"
echo "   finger jlundsgaard@wisc.edu"
finger jlundsgaard@wisc.edu 2>/dev/null || echo "   (finger 服务不可用)"
echo ""

echo "2. 查看用户账户信息"
echo "   getent passwd jlundsgaard"
getent passwd jlundsgaard 2>/dev/null || echo "   (getent 不可用)"
echo ""

echo "3. 查看 ivf 目录是否有 README 或联系信息"
echo ""
echo "查找 README 文件："
find /staging/groups/bhaskar_group/ivf/ -maxdepth 1 -name "README*" -o -name "*.txt" -o -name "*.md" 2>/dev/null | head -10

echo ""
echo "4. 查看 ivf 目录的所有者信息"
ls -ld /staging/groups/bhaskar_group/ivf/ 2>/dev/null

echo ""
echo "5. 查找 Git 仓库中的作者信息（如果有）"
if [ -d "/staging/groups/bhaskar_group/ivf/.git" ]; then
    echo "   找到 .git 目录，查看作者："
    cd /staging/groups/bhaskar_group/ivf/
    git log --format='%an <%ae>' | sort -u | head -5 2>/dev/null
    cd - > /dev/null
else
    echo "   (未找到 .git 目录)"
fi

echo ""
echo "============================================================"
echo "如果没有找到邮箱，可以尝试："
echo "  - jlundsgaard@wisc.edu (UW-Madison 邮箱格式)"
echo "  - 通过 CHTC 支持: chtc@cs.wisc.edu"
echo "  - 通过实验室 Slack/邮件列表"
echo "============================================================"






