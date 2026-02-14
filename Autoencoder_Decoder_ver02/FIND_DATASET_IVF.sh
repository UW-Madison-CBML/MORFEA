#!/bin/bash
# Find dataset_ivf.py in the repository

echo "=== Searching for dataset_ivf.py ==="
echo ""

# Search in repo
echo "Searching in ~/ivf..."
find ~/ivf -name "dataset_ivf.py" 2>/dev/null

echo ""
echo "=== Repository Structure ==="
ls -la ~/ivf/ | head -20
echo ""

if [ -d ~/ivf/Code ]; then
    echo "=== Files in ~/ivf/Code ==="
    ls -la ~/ivf/Code/ | head -20
fi

echo ""
echo "=== Alternative: Download from GitHub ==="
echo "If not found, download directly:"
echo "  curl -L -o dataset_ivf.py 'https://raw.githubusercontent.com/UW-Madison-CBML/ivf/main/Code/dataset_ivf.py'"





