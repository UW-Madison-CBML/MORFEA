#!/bin/bash
# Setup curvature analysis in group directory

echo "=== Setting up Curvature Analysis in Group Directory ==="
echo ""

GROUP_BASE="/staging/groups/bhaskar_group/rho9"
ANALYSIS_DIR="$GROUP_BASE/ivf_analysis"
SCRIPTS_DIR="$ANALYSIS_DIR/scripts"
OUTPUT_DIR="$GROUP_BASE/curvature_analysis"

echo "1. Creating directories..."
mkdir -p "$SCRIPTS_DIR"
mkdir -p "$OUTPUT_DIR"
echo "   ✓ Created: $SCRIPTS_DIR"
echo "   ✓ Created: $OUTPUT_DIR"
echo ""

echo "2. Checking if script exists locally..."
SCRIPT_PATH="$(dirname "$0")/analyze_trajectory_curvature.py"
if [ -f "$SCRIPT_PATH" ]; then
    echo "   ✓ Found: $SCRIPT_PATH"
    echo "   Copying to group directory..."
    cp "$SCRIPT_PATH" "$SCRIPTS_DIR/"
    echo "   ✓ Copied to: $SCRIPTS_DIR/analyze_trajectory_curvature.py"
else
    echo "   ⚠️  Script not found at: $SCRIPT_PATH"
    echo "   Please upload manually:"
    echo "     scp scripts/analyze_trajectory_curvature.py rho9@ap2001.chtc.wisc.edu:$SCRIPTS_DIR/"
fi

echo ""
echo "3. Directory structure:"
echo "   $GROUP_BASE/"
echo "   ├── ivf_data/                    # Dataset"
echo "   ├── ivf_analysis/                # Scripts"
echo "   │   └── scripts/"
echo "   │       └── analyze_trajectory_curvature.py"
echo "   └── curvature_analysis/          # Results (auto-created)"
echo ""

echo "4. To run analysis:"
echo "   cd $ANALYSIS_DIR"
echo "   python3 scripts/analyze_trajectory_curvature.py --video_name ZS435-5"
echo ""

echo "5. Results will be saved to:"
echo "   $OUTPUT_DIR/"
echo ""

echo "✅ Setup complete!"

