#!/bin/bash

quota -s

echo ""
echo ""
du -sh /home/rho9

echo ""
du -sh /staging/groups/bhaskar_group/rho9

echo ""
HOME_SIZE=$(du -sm /home/rho9 2>/dev/null | awk '{print $1}')
STAGING_SIZE=$(du -sm /staging/groups/bhaskar_group/rho9 2>/dev/null | awk '{print $1}')
TOTAL=$((HOME_SIZE + STAGING_SIZE))
QUOTA_USED=$(quota -s 2>/dev/null | grep "/dev/md9" | awk '{print $2}' | sed 's/M//')
echo "  Home: ${HOME_SIZE}M"
echo "  Staging/rho9: ${STAGING_SIZE}M"

echo ""
du -sh /staging/groups/bhaskar_group/ivf/ 2>/dev/null | head -1






