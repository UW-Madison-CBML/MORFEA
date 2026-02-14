#!/bin/bash
# Check staging directory and identify old/incomplete files to clean up

set -e

echo "============================================================"
echo "Staging Directory Check and Cleanup"
echo "============================================================"
echo ""

# Define staging paths
STAGING_BASE="/staging/groups/bhaskar_group/rho9"
IVF_DATA_DIR="$STAGING_BASE/ivf_data"
IVF_ANALYSIS_DIR="$STAGING_BASE/ivf_analysis"

echo "Checking staging directories..."
echo ""

# Check ivf_data directory
echo "============================================================"
echo "1. IVF Data Directory: $IVF_DATA_DIR"
echo "============================================================"
if [ -d "$IVF_DATA_DIR" ]; then
    echo "✓ Directory exists"
    echo ""
    echo "Contents:"
    ls -lh "$IVF_DATA_DIR" 2>/dev/null | head -20
    echo ""
    
    # Check for tar.gz
    if [ -f "$IVF_DATA_DIR/embryo_dataset.tar.gz" ]; then
        TAR_SIZE=$(du -sh "$IVF_DATA_DIR/embryo_dataset.tar.gz" | cut -f1)
        echo "✓ Found tar.gz: $TAR_SIZE"
    else
        echo "⚠️  No tar.gz file found"
    fi
    echo ""
    
    # Check for extracted dataset
    if [ -d "$IVF_DATA_DIR/embryo_dataset" ]; then
        EXTRACTED_SIZE=$(du -sh "$IVF_DATA_DIR/embryo_dataset" 2>/dev/null | cut -f1)
        EXTRACTED_COUNT=$(ls "$IVF_DATA_DIR/embryo_dataset" 2>/dev/null | wc -l)
        echo "✓ Found extracted dataset:"
        echo "   Size: $EXTRACTED_SIZE"
        echo "   Cell directories: $EXTRACTED_COUNT"
        
        # Check if it looks complete (should be ~12GB and have many cells)
        if [ "$EXTRACTED_COUNT" -lt 10 ]; then
            echo "   ⚠️  WARNING: Very few cell directories - might be incomplete!"
        fi
    else
        echo "⚠️  No extracted dataset directory found"
    fi
    echo ""
    
    # Check for other files/directories
    echo "Other files/directories in ivf_data:"
    ls -lh "$IVF_DATA_DIR" 2>/dev/null | grep -v "embryo_dataset" | grep -v "total" | head -10
    echo ""
else
    echo "❌ Directory does not exist: $IVF_DATA_DIR"
fi

# Check ivf_analysis directory
echo "============================================================"
echo "2. IVF Analysis Directory: $IVF_ANALYSIS_DIR"
echo "============================================================"
if [ -d "$IVF_ANALYSIS_DIR" ]; then
    echo "✓ Directory exists"
    echo ""
    echo "Contents:"
    ls -lh "$IVF_ANALYSIS_DIR" 2>/dev/null | head -20
    echo ""
    
    # Check subdirectories
    if [ -d "$IVF_ANALYSIS_DIR/scripts" ]; then
        SCRIPT_COUNT=$(ls "$IVF_ANALYSIS_DIR/scripts" 2>/dev/null | wc -l)
        echo "✓ scripts/ directory: $SCRIPT_COUNT files"
    fi
    
    if [ -d "$IVF_ANALYSIS_DIR/curvature_analysis" ]; then
        CURV_SIZE=$(du -sh "$IVF_ANALYSIS_DIR/curvature_analysis" 2>/dev/null | cut -f1)
        echo "✓ curvature_analysis/ directory: $CURV_SIZE"
    fi
    echo ""
else
    echo "⚠️  Directory does not exist: $IVF_ANALYSIS_DIR"
fi

# Check for other rho9 directories
echo "============================================================"
echo "3. Other Directories in rho9 Staging"
echo "============================================================"
if [ -d "$STAGING_BASE" ]; then
    echo "All directories in $STAGING_BASE:"
    ls -lh "$STAGING_BASE" 2>/dev/null | head -20
    echo ""
    
    # Calculate total size
    TOTAL_SIZE=$(du -sh "$STAGING_BASE" 2>/dev/null | cut -f1)
    echo "Total size of rho9 staging: $TOTAL_SIZE"
    echo ""
fi

# Summary and recommendations
echo "============================================================"
echo "Summary and Recommendations"
echo "============================================================"
echo ""

# Check if there's an incomplete extraction in home directory
HOME_EXTRACT="$HOME/ivf_repo/data_raw/embryo_dataset"
if [ -d "$HOME_EXTRACT" ]; then
    HOME_SIZE=$(du -sh "$HOME_EXTRACT" 2>/dev/null | cut -f1)
    HOME_COUNT=$(ls "$HOME_EXTRACT" 2>/dev/null | wc -l)
    echo "⚠️  Found incomplete extraction in HOME directory:"
    echo "   $HOME_EXTRACT"
    echo "   Size: $HOME_SIZE"
    echo "   Cell directories: $HOME_COUNT"
    echo "   → Consider deleting this to free up home quota"
    echo ""
fi

# Recommendations
echo "Recommendations:"
echo "1. If embryo_dataset in staging is incomplete (< 10 cells), delete it and re-extract"
echo "2. Delete any incomplete extractions from home directory"
echo "3. Keep the tar.gz file in staging (needed for re-extraction)"
echo ""

# Interactive cleanup option
read -p "Do you want to see cleanup commands? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "============================================================"
    echo "Cleanup Commands (run manually if needed)"
    echo "============================================================"
    echo ""
    
    # Check for incomplete staging extraction
    if [ -d "$IVF_DATA_DIR/embryo_dataset" ]; then
        EXTRACTED_COUNT=$(ls "$IVF_DATA_DIR/embryo_dataset" 2>/dev/null | wc -l)
        if [ "$EXTRACTED_COUNT" -lt 10 ]; then
            echo "# Remove incomplete staging extraction:"
            echo "rm -rf $IVF_DATA_DIR/embryo_dataset"
            echo ""
        fi
    fi
    
    # Check for home directory extraction
    if [ -d "$HOME_EXTRACT" ]; then
        echo "# Remove incomplete home extraction:"
        echo "rm -rf $HOME_EXTRACT"
        echo ""
    fi
    
    echo "# To extract properly to staging:"
    echo "cd $IVF_DATA_DIR"
    echo "tar -xzvf embryo_dataset.tar.gz"
    echo ""
fi

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"

