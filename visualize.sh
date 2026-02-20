#!/bin/bash

# Default values
OUTPUT_DIR="$1_plots"
BATCH_SIZE=50
GRADES_FILE="embryo_dataset_grades.csv"

# Display usage information
usage() {
    cat << EOF
Usage: $0 <latents_csv> [OPTIONS]

Options:
    --all                   Process all cells in the CSV
    --cells "cell1,cell2"   Process specific cells (comma-separated)
    --grade-filter FILTER   Filter by grade (e.g., "A", "B", "any_B", "A-A")
    --by-grade             Organize plots into subdirectories by grade
    --create-merged        Create merged comparison plot
    --output DIR           Output directory (default: plots)
    --batch-size N         Process N cells per batch (default: 50)
    --grades-file FILE     Path to grades CSV (default: embryo_dataset_grades.csv)

Examples:
    # Process all cells
    $0 latents.csv --all

    # Process specific cells
    $0 latents.csv --cells "cell1,cell2,cell3"

    # Process all B-grade cells with merged plot
    $0 latents.csv --grade-filter "any_B" --by-grade --create-merged

    # Process A-A grade cells
    $0 latents.csv --grade-filter "A-A" --by-grade --output aa_plots

    # Legacy: process specific cells (old interface)
    $0 latents.csv "cell1,cell2,cell3"
EOF
    exit 1
}

# Check if required arguments are provided
if [ $# -lt 1 ]; then
    usage
fi

LATENTS_CSV=$1
shift

# Build the command
CMD="python visualize.py \"$LATENTS_CSV\""

# Parse arguments
LEGACY_MODE=false
while [ $# -gt 0 ]; do
    case "$1" in
        --all)
            CMD="$CMD --all"
            shift
            ;;
        --cells)
            CMD="$CMD --cells \"$2\""
            shift 2
            ;;
        --grade-filter)
            CMD="$CMD --grade-filter \"$2\""
            shift 2
            ;;
        --by-grade)
            CMD="$CMD --by-grade"
            shift
            ;;
        --create-merged)
            CMD="$CMD --create-merged"
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            CMD="$CMD --output \"$2\""
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            CMD="$CMD --batch-size $2"
            shift 2
            ;;
        --grades-file)
            GRADES_FILE="$2"
            CMD="$CMD --grades-file \"$2\""
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            # Legacy mode: positional argument for cell_line
            LEGACY_MODE=true
            CMD="$CMD \"$1\""
            shift
            ;;
    esac
done

# Check if plots.tar.gz exists - extract if present, create if not
if [ -f "plots.tar.gz" ]; then
    echo "Extracting existing plots..."
    tar -xzvf plots.tar.gz
else
    echo "Creating new plots directory..."
    mkdir -p "$OUTPUT_DIR"
fi

# Extract annotations if available
if [ -f "embryo_dataset_annotations.tar.gz" ]; then
    echo "Extracting embryo annotations..."
    tar -xzvf embryo_dataset_annotations.tar.gz
fi

tar -xzvf latents.tar.gz
# Run visualization
echo "Running: $CMD"
eval $CMD
# Check if visualization succeeded
if [ $? -eq 0 ]; then
    # Rezip the plots
    echo "Compressing plots..."
    tar -czvf "${1}"_plots.tar.gz "$OUTPUT_DIR"
    echo "Plots saved to ${1}_plots.tar.gz"
else
    echo "Error: Visualization failed"
    exit 1
fi
ls -lh
