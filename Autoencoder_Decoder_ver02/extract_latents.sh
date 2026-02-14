#!/bin/bash
# extract_latents.sh
# Wrapper script for extracting latent trajectories on CHTC
# This script copies files from staging to working directory

set -e

echo "=== Extract Latent Trajectories on CHTC ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Staging directory
STAGING_DIR="/staging/groups/bhaskar_group/rho9"

# Parse arguments
CHECKPOINT_ARG="$1"
MODEL_VERSION="$2"

if [ -z "$CHECKPOINT_ARG" ] || [ -z "$MODEL_VERSION" ]; then
    echo "Usage: $0 <checkpoint_path> <model_version_name>"
    exit 1
fi

echo "Checkpoint argument: $CHECKPOINT_ARG"
echo "Model version: $MODEL_VERSION"
echo ""

# Copy files from staging to working directory
echo "Copying files from staging to working directory..."
cp "$STAGING_DIR/extract_all_latent_trajectories.py" ./
cp "$STAGING_DIR/model.py" ./
cp "$STAGING_DIR/dataset_ivf.py" ./
cp "$STAGING_DIR/build_index.py" ./
cp "$STAGING_DIR/index.csv" ./

# Handle checkpoint path
if [[ "$CHECKPOINT_ARG" == /* ]]; then
    # Absolute path - copy from staging
    if [ ! -f "$CHECKPOINT_ARG" ]; then
        echo "Error: Checkpoint not found: $CHECKPOINT_ARG"
        exit 1
    fi
    cp "$CHECKPOINT_ARG" ./checkpoint.pt
    CHECKPOINT="./checkpoint.pt"
else
    # Relative path - should be in current directory after transfer
    CHECKPOINT="$CHECKPOINT_ARG"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint not found: $CHECKPOINT"
        exit 1
    fi
fi

echo "✓ Files copied successfully"
echo ""

# Setup Python environment
echo "Setting up Python environment..."
python3 --version

# Check and install missing packages if needed
echo "Checking Python packages..."
python3 << 'PYTHON_EOF'
import sys
import subprocess

def check_package(name, import_name=None):
    """Check if a package is available, return (available, package_name)"""
    if import_name is None:
        import_name = name
    try:
        __import__(import_name)
        return True, name
    except ImportError:
        return False, name

# Check all packages
packages = [
    ("torch", "torch"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("Pillow", "PIL")
]

missing = []
for pkg_name, import_name in packages:
    available, _ = check_package(pkg_name, import_name)
    if available:
        print(f"  ✓ {pkg_name}")
    else:
        missing.append(pkg_name)
        print(f"  ✗ {pkg_name} - MISSING")

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    print("Attempting to install (this may take a few minutes)...")
    
    # Install all missing packages in one command (faster and more reliable)
    pip_packages = [pkg.lower() if pkg == "Pillow" else pkg for pkg in missing]
    print(f"  Installing: {', '.join(pip_packages)}")
    
    try:
        # Install all at once with timeout
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--user", "--no-cache-dir"] + pip_packages,
            timeout=600,  # 10 minute timeout for all packages
            stdout=subprocess.DEVNULL,  # Suppress verbose output
            stderr=subprocess.PIPE
        )
        print("✓ Installed successfully")
        
        # Re-check after installation
        print("\nRe-checking installed packages...")
        all_ok = True
        for pkg_name, import_name in packages:
            available, _ = check_package(pkg_name, import_name)
            if available:
                print(f"  ✓ {pkg_name}")
            else:
                print(f"  ✗ {pkg_name} - STILL MISSING")
                all_ok = False
        
        if not all_ok:
            print("\n⚠️  Warning: Some packages still missing after installation")
            print("This may be due to Python path issues. Continuing anyway...")
    except subprocess.TimeoutExpired:
        print("  ✗ Installation timed out")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Installation failed: {e}")
        sys.exit(1)
else:
    print("✓ All required packages available")
PYTHON_EOF

# Check if index.csv exists
if [ ! -f "index.csv" ]; then
    echo "Error: index.csv not found after copy"
    exit 1
fi

echo "Detecting GPU (for logging, Python will auto-detect)..."
python3 << 'PYTHON_EOF'
import torch
if torch.cuda.is_available():
    print("GPU detected via PyTorch")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("  (Python script will auto-detect and use CUDA)")
else:
    print("No GPU detected")
    print("  (Python script will auto-detect and use CPU)")
PYTHON_EOF

# Run extraction
echo ""
echo "Running extraction..."
# Add --max_embryos parameter if needed (uncomment and set number)
# MAX_EMBRYOS=""  # Process all embryos
MAX_EMBRYOS=""  # Set to number like "10" to limit, or "" for all

# Check if files exist before running
echo "Checking files..."
ls -la extract_all_latent_trajectories.py model.py dataset_ivf.py build_index.py index.csv
echo ""

# Pre-create output directory structure to ensure it exists
# This helps with debugging and ensures the directory structure is ready
OUTPUT_BASE="model_latents"
OUTPUT_DIR="$OUTPUT_BASE/$MODEL_VERSION"
LATENTS_DIR="$OUTPUT_DIR/latents"

echo "Pre-creating output directory structure..."
mkdir -p "$LATENTS_DIR"
echo "✓ Output directory structure created: $OUTPUT_DIR"
echo "  - Base: $OUTPUT_BASE"
echo "  - Model version: $OUTPUT_DIR"
echo "  - Latents: $LATENTS_DIR"
echo ""

# Run with error handling
set +e  # Don't exit on error, so we can see the full error message
echo "Starting Python script at $(date)..." | tee -a extraction_progress.log
if [ -n "$MAX_EMBRYOS" ]; then
    python3 -u extract_all_latent_trajectories.py \
        --checkpoint "$CHECKPOINT" \
        --model_version "$MODEL_VERSION" \
        --index_csv index.csv \
        --max_embryos "$MAX_EMBRYOS" \
        --output_dir "$OUTPUT_BASE" 2>&1 | tee -a extraction_progress.log
    EXIT_CODE=$?
else
    python3 -u extract_all_latent_trajectories.py \
        --checkpoint "$CHECKPOINT" \
        --model_version "$MODEL_VERSION" \
        --index_csv index.csv \
        --output_dir "$OUTPUT_BASE" 2>&1 | tee -a extraction_progress.log
    EXIT_CODE=$?
fi
echo "Python script finished at $(date) with exit code $EXIT_CODE" | tee -a extraction_progress.log
set -e

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Python script failed with exit code $EXIT_CODE"
    echo "Check the output above for error details"
    exit $EXIT_CODE
fi

echo ""
echo "=== Extraction Complete ==="
echo "Checking results..."
echo "Results should be in: $OUTPUT_DIR/"
if [ -d "$OUTPUT_DIR" ]; then
    echo "✓ Output directory exists"
    ls -lh "$OUTPUT_DIR/"
    echo ""
    if [ -d "$LATENTS_DIR" ]; then
        echo "✓ Latents directory exists"
        file_count=$(ls -1 "$LATENTS_DIR"/*.npy 2>/dev/null | wc -l)
        echo "  Found $file_count .npy files"
        if [ $file_count -gt 0 ]; then
            ls -lh "$LATENTS_DIR"/*.npy | head -5
        fi
    else
        echo "⚠️  Latents directory not found: $LATENTS_DIR"
    fi
    
    # Check if metadata exists
    if [ -f "$OUTPUT_DIR/metadata.json" ]; then
        echo ""
        echo "✓ Metadata file exists"
        echo "  Preview:"
        head -20 "$OUTPUT_DIR/metadata.json"
    else
        echo ""
        echo "⚠️  Metadata file not found"
    fi
else
    echo "❌ Output directory not found: $OUTPUT_DIR"
    echo "Current directory contents:"
    ls -la
    echo ""
    echo "Looking for model_latents directory:"
    find . -name "model_latents" -type d 2>/dev/null || echo "Not found"
fi

