#!/bin/bash
# Clean up incomplete extraction and re-extract properly to staging

set -e

echo "============================================================"
echo "Cleanup and Re-extract Dataset to Staging"
echo "============================================================"
echo ""

# Define paths
TAR_FILE="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"
STAGING_DIR="/staging/groups/bhaskar_group/rho9/ivf_data"
EXTRACT_DIR="$STAGING_DIR/embryo_dataset"

# Step 1: Check current state
echo "Step 1: Checking current state..."
echo ""

if [ -d "$EXTRACT_DIR" ]; then
    CURRENT_SIZE=$(du -sh "$EXTRACT_DIR" 2>/dev/null | cut -f1)
    CURRENT_COUNT=$(ls -d "$EXTRACT_DIR"/*/ 2>/dev/null | wc -l)
    echo "Current extraction:"
    echo "   Size: $CURRENT_SIZE"
    echo "   Cell directories: $CURRENT_COUNT"
    echo ""
    
    if [ "$CURRENT_COUNT" -lt 10 ]; then
        echo "⚠️  INCOMPLETE extraction detected!"
        echo "   This needs to be removed and re-extracted."
        echo ""
    fi
else
    echo "✓ No existing extraction directory"
    echo ""
fi

# Step 2: Check tar.gz file
if [ ! -f "$TAR_FILE" ]; then
    echo "❌ ERROR: Tar.gz file not found:"
    echo "   $TAR_FILE"
    exit 1
fi

TAR_SIZE=$(du -sh "$TAR_FILE" | cut -f1)
echo "✓ Found tar.gz file: $TAR_SIZE"
echo ""

# Step 3: Check available space
echo "Step 2: Checking available space..."
STAGING_AVAIL=$(df -h "$STAGING_DIR" | tail -1 | awk '{print $4}')
echo "   Available space in staging: $STAGING_AVAIL"
echo ""

# Step 4: Remove incomplete extraction
if [ -d "$EXTRACT_DIR" ]; then
    echo "Step 3: Removing incomplete extraction..."
    echo "   Removing: $EXTRACT_DIR"
    rm -rf "$EXTRACT_DIR"
    echo "   ✓ Removed"
    echo ""
fi

# Step 5: Extract to staging
echo "============================================================"
echo "Step 4: Extracting to staging directory..."
echo "============================================================"
echo "Source: $TAR_FILE"
echo "Destination: $EXTRACT_DIR"
echo ""
echo "⚠️  This will take 10-30 minutes depending on disk speed"
echo "⚠️  Make sure you're extracting to STAGING, not HOME!"
echo ""

# CRITICAL: Use -C flag to extract directly to staging directory
# This ensures we extract to the right location
cd "$STAGING_DIR"
echo "Current directory: $(pwd)"
echo "Extracting..."
echo ""

# Extract with verbose output (you can see progress)
tar -xzvf "$TAR_FILE"

echo ""
echo "============================================================"
echo "Step 5: Verifying extraction..."
echo "============================================================"
echo ""

if [ -d "$EXTRACT_DIR" ]; then
    FINAL_SIZE=$(du -sh "$EXTRACT_DIR" | cut -f1)
    FINAL_COUNT=$(ls -d "$EXTRACT_DIR"/*/ 2>/dev/null | wc -l)
    echo "✓ Extraction complete!"
    echo "   Location: $EXTRACT_DIR"
    echo "   Size: $FINAL_SIZE"
    echo "   Cell directories: $FINAL_COUNT"
    echo ""
    
    if [ "$FINAL_COUNT" -gt 100 ]; then
        echo "✅ SUCCESS! Extraction looks complete!"
    elif [ "$FINAL_COUNT" -gt 10 ]; then
        echo "⚠️  Partial extraction? Check if more cells should exist."
    else
        echo "❌ Still incomplete! Check for errors above."
    fi
    echo ""
    
    echo "First 10 cells:"
    ls -d "$EXTRACT_DIR"/*/ 2>/dev/null | head -10 | xargs -n1 basename
    echo ""
    echo "✅ Dataset ready for use!"
else
    echo "❌ ERROR: Extraction directory not found after extraction"
    exit 1
fi

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"

