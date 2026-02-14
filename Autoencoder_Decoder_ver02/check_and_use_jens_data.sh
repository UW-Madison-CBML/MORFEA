#!/bin/bash
# Check Jens's dataset and set it up correctly
# This will diagnose why extraction failed and fix it

echo "=== Checking Jens's Dataset ==="
echo ""

JENS_BASE="/staging/groups/bhaskar_group/ivf"
JENS_TAR="$JENS_BASE/embryo_dataset.tar.gz"
JENS_DIR="$JENS_BASE/embryo_dataset"

echo "1. Checking Jens's dataset location..."
echo ""

# Check if directory exists (already extracted)
if [ -d "$JENS_DIR" ]; then
    echo "✅ Found extracted dataset directory:"
    echo "   Path: $JENS_DIR"
    SIZE=$(du -sh "$JENS_DIR" 2>/dev/null | cut -f1)
    CELL_COUNT=$(find "$JENS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    FILE_COUNT=$(find "$JENS_DIR" -type f \( -name "*.jpeg" -o -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
    echo "   Size: $SIZE"
    echo "   Cell directories: $CELL_COUNT"
    echo "   Image files: $FILE_COUNT"
    echo ""
    
    if [ "$CELL_COUNT" -gt 0 ] && [ "$FILE_COUNT" -gt 1000 ]; then
        echo "✅ Dataset looks complete!"
        USE_DIRECTORY=true
    else
        echo "⚠️  Dataset seems incomplete or empty"
        USE_DIRECTORY=false
    fi
else
    echo "❌ Extracted directory not found: $JENS_DIR"
    USE_DIRECTORY=false
fi

# Check if tar.gz exists
if [ -f "$JENS_TAR" ]; then
    echo "✅ Found tar.gz file:"
    echo "   Path: $JENS_TAR"
    TAR_SIZE=$(du -sh "$JENS_TAR" 2>/dev/null | cut -f1)
    echo "   Size: $TAR_SIZE"
    echo ""
    USE_TAR=true
else
    echo "❌ Tar.gz file not found: $JENS_TAR"
    USE_TAR=false
fi

echo ""
echo "2. Checking permissions..."
if [ -d "$JENS_DIR" ]; then
    if [ -r "$JENS_DIR" ]; then
        echo "✅ Read permission: OK"
    else
        echo "❌ Read permission: DENIED"
    fi
    
    if [ -x "$JENS_DIR" ]; then
        echo "✅ Execute permission: OK"
    else
        echo "❌ Execute permission: DENIED"
    fi
fi

echo ""
echo "3. Sample cell directories (first 5):"
if [ -d "$JENS_DIR" ]; then
    find "$JENS_DIR" -mindepth 1 -maxdepth 1 -type d | head -5 | while read dir; do
        cell_name=$(basename "$dir")
        file_count=$(find "$dir" -type f \( -name "*.jpeg" -o -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
        echo "   $cell_name: $file_count files"
    done
fi

echo ""
echo "=== Setup Recommendation ==="
echo ""

if [ "$USE_DIRECTORY" = true ]; then
    echo "✅ Use existing extracted directory"
    echo ""
    echo "To set up:"
    echo "  cd ~/ivf_repo"
    echo "  rm -f data"
    echo "  ln -s $JENS_DIR data"
    echo "  ls -la data"
    echo ""
    read -p "Set up symlink now? (y/n): " setup
    if [ "$setup" = "y" ]; then
        cd ~/ivf_repo 2>/dev/null || cd "$HOME"
        if [ -d "ivf_repo" ]; then
            cd ivf_repo
            rm -f data
            ln -s "$JENS_DIR" data
            echo "✅ Symlink created: data -> $JENS_DIR"
            ls -la data
        else
            echo "❌ ~/ivf_repo not found"
        fi
    fi
elif [ "$USE_TAR" = true ]; then
    echo "⚠️  Need to extract from tar.gz"
    echo ""
    echo "To extract:"
    echo "  mkdir -p ~/ivf_repo/data_raw"
    echo "  cd ~/ivf_repo/data_raw"
    echo "  tar -xzvf $JENS_TAR 2>&1 | tee extraction.log"
    echo ""
    echo "Then create symlink:"
    echo "  cd ~/ivf_repo"
    echo "  ln -s data_raw/embryo_dataset data"
    echo ""
    read -p "Extract now? (y/n): " extract
    if [ "$extract" = "y" ]; then
        mkdir -p ~/ivf_repo/data_raw
        cd ~/ivf_repo/data_raw
        echo "Extracting (this may take 10-30 minutes)..."
        tar -xzvf "$JENS_TAR" 2>&1 | tee extraction.log
        if [ $? -eq 0 ]; then
            echo "✅ Extraction complete!"
            cd ~/ivf_repo
            rm -f data
            ln -s data_raw/embryo_dataset data
            echo "✅ Symlink created"
        else
            echo "❌ Extraction failed - check extraction.log"
        fi
    fi
else
    echo "❌ Cannot find Jens's dataset"
    echo "   Checked:"
    echo "     - Directory: $JENS_DIR"
    echo "     - Tar.gz: $JENS_TAR"
    echo ""
    echo "   Please verify the path with Jens"
fi

