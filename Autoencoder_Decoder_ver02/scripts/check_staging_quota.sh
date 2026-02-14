#!/bin/bash
# Check staging quota and verify extraction location

set -e

echo "============================================================"
echo "Staging Quota and Space Check"
echo "============================================================"
echo ""

# Check staging filesystem info
STAGING_DIR="/staging/groups/bhaskar_group/rho9"

echo "1. Checking staging filesystem space:"
df -h "$STAGING_DIR" | head -2
df -h "$STAGING_DIR" | tail -1
echo ""

# Check if there's a quota command
echo "2. Checking quota (if available):"
if command -v quota &> /dev/null; then
    quota -s 2>/dev/null || echo "   (quota command not available or no quota set)"
else
    echo "   (quota command not found)"
fi
echo ""

# Check current usage in staging
echo "3. Current staging directory usage:"
echo "   Location: $STAGING_DIR"
if [ -d "$STAGING_DIR" ]; then
    du -sh "$STAGING_DIR" 2>/dev/null || echo "   (cannot calculate size)"
    echo ""
    echo "   Breakdown:"
    for dir in "$STAGING_DIR"/*; do
        if [ -d "$dir" ]; then
            dir_name=$(basename "$dir")
            dir_size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo "     $dir_name: $dir_size"
        fi
    done
else
    echo "   (directory does not exist)"
fi
echo ""

# Check where tar is trying to extract
echo "4. Checking extraction target:"
TAR_FILE="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"
EXTRACT_DIR="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset"

if [ -f "$TAR_FILE" ]; then
    echo "   ✓ Tar.gz file exists: $TAR_FILE"
    TAR_SIZE=$(du -sh "$TAR_FILE" | cut -f1)
    echo "   Size: $TAR_SIZE"
else
    echo "   ❌ Tar.gz file not found: $TAR_FILE"
fi
echo ""

echo "   Extraction target: $EXTRACT_DIR"
if [ -d "$EXTRACT_DIR" ]; then
    EXTRACT_SIZE=$(du -sh "$EXTRACT_DIR" 2>/dev/null | cut -f1)
    EXTRACT_COUNT=$(ls -d "$EXTRACT_DIR"/*/ 2>/dev/null | wc -l)
    echo "   Current size: $EXTRACT_SIZE"
    echo "   Cell directories: $EXTRACT_COUNT"
else
    echo "   (does not exist yet)"
fi
echo ""

# Check if there's a group quota
echo "5. Checking for group quota information:"
echo "   (Group: bhaskar_group)"
echo "   (This might have its own quota limits)"
echo ""

# Recommendations
echo "============================================================"
echo "Recommendations"
echo "============================================================"
echo ""

# Check available space
AVAIL_SPACE=$(df -h "$STAGING_DIR" | tail -1 | awk '{print $4}')
echo "Available space: $AVAIL_SPACE"
echo ""

if [ -f "$TAR_FILE" ]; then
    TAR_SIZE_GB=$(du -sm "$TAR_FILE" 2>/dev/null | cut -f1)
    TAR_SIZE_GB=$((TAR_SIZE_GB / 1024))
    echo "Tar.gz size: ~${TAR_SIZE_GB}GB"
    echo "Extracted size will be: ~${TAR_SIZE_GB}GB"
    echo ""
    
    echo "If you're hitting quota:"
    echo "1. Check if there's a group quota limit"
    echo "2. Contact CHTC support about group quota"
    echo "3. Consider extracting in smaller chunks (not recommended)"
    echo "4. Check if other group members are using space"
fi

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"

