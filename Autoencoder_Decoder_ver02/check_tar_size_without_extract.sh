#!/bin/bash
# Check tar.gz uncompressed size without extracting

echo "=== Check Tar.gz Uncompressed Size (Without Extracting) ==="
echo ""

TAR_FILE="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"

if [ ! -f "$TAR_FILE" ]; then
    echo "❌ File not found: $TAR_FILE"
    exit 1
fi

echo "Tar.gz file: $TAR_FILE"
COMPRESSED_SIZE=$(du -sh "$TAR_FILE" | cut -f1)
COMPRESSED_SIZE_BYTES=$(du -sb "$TAR_FILE" | cut -f1)
COMPRESSED_SIZE_GB=$(echo "scale=2; $COMPRESSED_SIZE_BYTES / 1024 / 1024 / 1024" | bc)
echo "Compressed size: $COMPRESSED_SIZE (${COMPRESSED_SIZE_GB} GB)"
echo ""

echo "Calculating uncompressed size..."
echo "(This may take a few minutes for large archives)"
echo ""

# Method 1: Use tar --list --verbose to get file sizes
# This lists all files with their sizes without extracting
TOTAL_BYTES=0
FILE_COUNT=0
CELL_DIRS=()

echo "Analyzing archive contents..."
while IFS= read -r line; do
    # Parse tar verbose output: permissions size date name
    # Format: -rw-r--r-- user/group size date time name
    if [[ $line =~ ^- ]]; then
        # Extract size (usually 5th field, but can vary)
        SIZE=$(echo "$line" | awk '{print $5}')
        if [[ $SIZE =~ ^[0-9]+$ ]]; then
            TOTAL_BYTES=$((TOTAL_BYTES + SIZE))
            FILE_COUNT=$((FILE_COUNT + 1))
            
            # Extract filename
            FILENAME=$(echo "$line" | awk '{for(i=6;i<=NF;i++) printf "%s ", $i; print ""}' | sed 's/ $//')
            
            # Check if it's a cell directory (first level directory)
            if [[ $FILENAME =~ ^([^/]+)/$ ]]; then
                CELL_NAME="${BASH_REMATCH[1]}"
                if [[ ! " ${CELL_DIRS[@]} " =~ " ${CELL_NAME} " ]]; then
                    CELL_DIRS+=("$CELL_NAME")
                fi
            fi
        fi
    fi
done < <(tar -tzv "$TAR_FILE" 2>/dev/null)

if [ $TOTAL_BYTES -eq 0 ]; then
    echo "⚠️  Could not calculate from tar listing, trying alternative method..."
    echo ""
    echo "Alternative: Using tar --to-stdout to estimate..."
    echo "(This will read the archive but not save files)"
    echo ""
    
    # Method 2: Use tar --to-stdout to /dev/null (just to verify it's readable)
    # But this doesn't give us size easily, so let's try a different approach
    
    # Method 3: Count files and estimate
    FILE_COUNT=$(tar -tzf "$TAR_FILE" 2>/dev/null | wc -l)
    echo "Total files in archive: $FILE_COUNT"
    
    # Get sample of file sizes
    echo "Sampling file sizes (first 100 files)..."
    SAMPLE_SIZE=0
    SAMPLE_COUNT=0
    while IFS= read -r line && [ $SAMPLE_COUNT -lt 100 ]; do
        if [[ $line =~ ^- ]]; then
            SIZE=$(echo "$line" | awk '{print $5}')
            if [[ $SIZE =~ ^[0-9]+$ ]]; then
                SAMPLE_SIZE=$((SAMPLE_SIZE + SIZE))
                SAMPLE_COUNT=$((SAMPLE_COUNT + 1))
            fi
        fi
    done < <(tar -tzv "$TAR_FILE" 2>/dev/null | head -100)
    
    if [ $SAMPLE_COUNT -gt 0 ]; then
        AVG_SIZE=$((SAMPLE_SIZE / SAMPLE_COUNT))
        ESTIMATED_TOTAL=$((AVG_SIZE * FILE_COUNT))
        ESTIMATED_GB=$(echo "scale=2; $ESTIMATED_TOTAL / 1024 / 1024 / 1024" | bc)
        echo "Estimated uncompressed size: ${ESTIMATED_GB} GB"
        TOTAL_BYTES=$ESTIMATED_TOTAL
    fi
fi

if [ $TOTAL_BYTES -gt 0 ]; then
    UNCOMPRESSED_GB=$(echo "scale=2; $TOTAL_BYTES / 1024 / 1024 / 1024" | bc)
    UNCOMPRESSED_MB=$(echo "scale=2; $TOTAL_BYTES / 1024 / 1024" | bc)
    
    echo ""
    echo "=== Results ==="
    echo "Compressed size: ${COMPRESSED_SIZE_GB} GB"
    echo "Uncompressed size: ${UNCOMPRESSED_GB} GB (${UNCOMPRESSED_MB} MB)"
    echo "File count: $FILE_COUNT"
    echo "Cell directories: ${#CELL_DIRS[@]}"
    echo ""
    
    if [ ${#CELL_DIRS[@]} -gt 0 ]; then
        echo "Cell directories found (first 10):"
        printf "  %s\n" "${CELL_DIRS[@]:0:10}"
    fi
    
    echo ""
    if (( $(echo "$UNCOMPRESSED_GB > 10" | bc -l) )); then
        echo "✅ Size looks correct (~12 GB expected)"
    else
        echo "⚠️  Size seems smaller than expected (${UNCOMPRESSED_GB} GB vs ~12 GB)"
    fi
    
    COMPRESSION_RATIO=$(echo "scale=2; $UNCOMPRESSED_GB / $COMPRESSED_SIZE_GB" | bc)
    echo "Compression ratio: ${COMPRESSION_RATIO}x"
else
    echo "❌ Could not calculate uncompressed size"
    echo "   Archive might be corrupted or in unexpected format"
fi

