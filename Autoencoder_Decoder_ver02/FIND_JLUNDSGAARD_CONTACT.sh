#!/bin/bash
# 查找 jlundsgaard 的联系方式

echo "=== 查找用户信息 ==="
getent passwd jlundsgaard 2>/dev/null || echo "无法获取用户信息"

echo ""
echo "=== 尝试 finger（如果有配置）==="
finger jlundsgaard 2>/dev/null || echo "finger 不可用"

echo ""
echo "=== 检查是否有其他联系方式 ==="
# 可以尝试查看 ivf 目录下是否有 README 或联系信息
if [ -f "/staging/groups/bhaskar_group/ivf/README*" ]; then
    echo "找到 README 文件："
    ls -la /staging/groups/bhaskar_group/ivf/README*
fi

echo ""
echo "=== 建议 ==="
echo "1. 查看用户信息：getent passwd jlundsgaard"
echo "2. 如果他在 Wisc.edu，邮箱可能是：jlundsgaard@wisc.edu"
echo "3. 或者通过实验室管理员/PI (Bhaskar) 联系"
echo "4. 或者联系 CHTC 支持：chtc@cs.wisc.edu"






