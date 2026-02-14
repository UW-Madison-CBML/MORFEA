#!/bin/bash

echo ""
du -sh ~/.cache

echo ""
du -sh ~/ivf-embryo-analysis-Raffael.tgz

echo ""
echo "   .venv: $(du -sh ~/.venv 2>/dev/null | awk '{print $1}')"
echo "   Desktop/.venv: $(du -sh ~/Desktop/.venv 2>/dev/null | awk '{print $1}')"
echo "   .local: $(du -sh ~/.local 2>/dev/null | awk '{print $1}')"

echo ""
echo ""
echo "
echo "rm -rf ~/.cache"
echo ""
echo "
echo "rm ~/ivf-embryo-analysis-Raffael.tgz"
echo ""
echo "
echo "
echo ""
echo "quota -s"






