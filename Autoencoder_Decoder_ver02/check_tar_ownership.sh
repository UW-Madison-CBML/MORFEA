#!/bin/bash
# Check who owns the tar.gz file

echo "=== Checking tar.gz File Ownership ==="
echo ""

YOUR_TAR="/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz"
JENS_TAR="/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz"

echo "1. Your tar.gz (rho9 path):"
if [ -f "$YOUR_TAR" ]; then
    echo "   ✅ Found: $YOUR_TAR"
    ls -lh "$YOUR_TAR"
    echo ""
    echo "   Owner: $(stat -c '%U' "$YOUR_TAR" 2>/dev/null || stat -f '%Su' "$YOUR_TAR" 2>/dev/null)"
    echo "   Group: $(stat -c '%G' "$YOUR_TAR" 2>/dev/null || stat -f '%Sg' "$YOUR_TAR" 2>/dev/null)"
    echo "   Date: $(stat -c '%y' "$YOUR_TAR" 2>/dev/null || stat -f '%Sm' "$YOUR_TAR" 2>/dev/null)"
    echo ""
    echo "   Path contains 'rho9' → This is YOUR tar.gz file"
else
    echo "   ❌ Not found: $YOUR_TAR"
fi

echo ""
echo "2. Jens's tar.gz (ivf path):"
if [ -f "$JENS_TAR" ]; then
    echo "   ✅ Found: $JENS_TAR"
    ls -lh "$JENS_TAR"
    echo ""
    echo "   Owner: $(stat -c '%U' "$JENS_TAR" 2>/dev/null || stat -f '%Su' "$JENS_TAR" 2>/dev/null)"
    echo "   Group: $(stat -c '%G' "$JENS_TAR" 2>/dev/null || stat -f '%Sg' "$JENS_TAR" 2>/dev/null)"
    echo "   Date: $(stat -c '%y' "$JENS_TAR" 2>/dev/null || stat -f '%Sm' "$JENS_TAR" 2>/dev/null)"
    echo ""
    echo "   Path contains 'ivf' (not rho9) → This is JENS's tar.gz file"
else
    echo "   ❌ Not found: $JENS_TAR"
fi

echo ""
echo "=== Summary ==="
echo ""

if [ -f "$YOUR_TAR" ]; then
    SIZE=$(du -sh "$YOUR_TAR" | cut -f1)
    echo "Your tar.gz: $YOUR_TAR ($SIZE)"
    echo "  → This is YOUR file (path contains 'rho9', owner is 'rho9')"
fi

if [ -f "$JENS_TAR" ]; then
    SIZE=$(du -sh "$JENS_TAR" | cut -f1)
    echo "Jens's tar.gz: $JENS_TAR ($SIZE)"
    echo "  → This is JENS's file (path contains 'ivf', not 'rho9')"
fi

