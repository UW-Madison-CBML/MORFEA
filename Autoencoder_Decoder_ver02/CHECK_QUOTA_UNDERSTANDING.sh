#!/bin/bash
# 检查配额使用情况

echo "=== 当前配额 ==="
quota -s

echo ""
echo "=== 检查各个目录占用配额的情况 ==="
echo ""
echo "1. Home 目录大小："
du -sh /home/rho9

echo ""
echo "2. Staging/rho9 目录大小："
du -sh /staging/groups/bhaskar_group/rho9

echo ""
echo "3. 两者总和（应该接近配额使用量）："
HOME_SIZE=$(du -sm /home/rho9 2>/dev/null | awk '{print $1}')
STAGING_SIZE=$(du -sm /staging/groups/bhaskar_group/rho9 2>/dev/null | awk '{print $1}')
TOTAL=$((HOME_SIZE + STAGING_SIZE))
QUOTA_USED=$(quota -s 2>/dev/null | grep "/dev/md9" | awk '{print $2}' | sed 's/M//')
echo "  Home: ${HOME_SIZE}M"
echo "  Staging/rho9: ${STAGING_SIZE}M"
echo "  总计: ${TOTAL}M"
echo "  配额显示: ${QUOTA_USED}M"

echo ""
echo "=== ivf 目录（不在你的配额中）==="
du -sh /staging/groups/bhaskar_group/ivf/ 2>/dev/null | head -1
echo "  （这个目录的大小不计入你的个人配额）"






