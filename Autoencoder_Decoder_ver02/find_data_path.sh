#!/bin/bash
# Find the correct data path on CHTC

echo ""

POSSIBLE_PATHS=(
    "/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset"
    "/staging/groups/bhaskar_group/ivf/embryo_dataset"
    "/staging/groups/bhaskar_group/ivf"
    "/staging/groups/bhaskar_group/rho9/ivf_data"
    "/project/bhaskar_group/ivf/embryo_dataset"
    "/project/bhaskar_group/ivf"
)

FOUND=false
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        if [ -d "$path/embryo_dataset" ]; then
            FOUND=true
            DATA_PATH="$path/embryo_dataset"
        elif [ "$(basename $path)" = "embryo_dataset" ]; then
            FOUND=true
            DATA_PATH="$path"
        else
            ls -d "$path"/*/ 2>/dev/null | head -3 | while read dir; do
                if [ -d "$dir" ] && [ -n "$(ls -A "$dir"/*.jpeg 2>/dev/null | head -1)" ]; then
                fi
            done
        fi
    else
    fi
done

echo ""
if [ "$FOUND" = true ]; then
    echo ""
    echo "  ln -sf $DATA_PATH data"
    echo ""
    echo "  python3 build_index.py --root $DATA_PATH --out index.csv"
else
    echo ""
    echo "  ls -la /staging/groups/bhaskar_group/"
    echo "  ls -la /staging/groups/bhaskar_group/rho9/"
    echo "  ls -la /staging/groups/bhaskar_group/ivf/"
fi
