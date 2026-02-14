#!/bin/bash
# Compare your dataset vs Jens's dataset

echo "=== Dataset Comparison ==="
echo ""

echo "1. Your dataset (rho9/ivf_data):"
YOUR_BASE="/staging/groups/bhaskar_group/rho9/ivf_data"
YOUR_TAR="$YOUR_BASE/embryo_dataset.tar.gz"
YOUR_DIR="$YOUR_BASE/embryo_dataset"

if [ -f "$YOUR_TAR" ]; then
    YOUR_SIZE=$(du -sh "$YOUR_TAR" | cut -f1)
    YOUR_SIZE_GB=$(echo "scale=2; $(du -sb "$YOUR_TAR" | cut -f1) / 1024 / 1024 / 1024" | bc)
    echo "   Tar.gz: $YOUR_SIZE (${YOUR_SIZE_GB} GB)"
    echo "   Type: Raw image dataset (original images)"
fi

if [ -d "$YOUR_DIR" ]; then
    YOUR_DIR_SIZE=$(du -sh "$YOUR_DIR" 2>/dev/null | cut -f1)
    YOUR_DIR_SIZE_GB=$(echo "scale=2; $(du -sb "$YOUR_DIR" 2>/dev/null | cut -f1) / 1024 / 1024 / 1024" | bc)
    CELL_COUNT=$(find "$YOUR_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "   Extracted: $YOUR_DIR_SIZE (${YOUR_DIR_SIZE_GB} GB, $CELL_COUNT cells)"
fi

echo ""
echo "2. Jens's dataset (ivf):"
JENS_BASE="/staging/groups/bhaskar_group/ivf"
JENS_SIZE=$(du -sh "$JENS_BASE" 2>/dev/null | cut -f1)
JENS_SIZE_GB=$(echo "scale=2; $(du -sb "$JENS_BASE" 2>/dev/null | cut -f1) / 1024 / 1024 / 1024" | bc)
echo "   Total: $JENS_SIZE (${JENS_SIZE_GB} GB)"
echo ""
echo "   Contents:"
echo "     - latents.csv: 2.6G (extracted latent vectors)"
echo "     - model_weights.pth: 291M (trained model)"
echo "     - annotations: 53K (metadata)"
echo "     - grades.csv: 10K (scores)"
echo ""
echo "   ⚠️  NO raw image dataset found"
echo "      (Only processed results, not original images)"

echo ""
echo "3. Comparison:"
echo ""
echo "   Your dataset:"
echo "     - Size: ~12 GB"
echo "     - Type: Raw image dataset"
echo "     - Usage: Can extract latents, train models, full pipeline"
echo ""
echo "   Jens's dataset:"
echo "     - Size: ~2.9 GB"
echo "     - Type: Processed results only (latents + model)"
echo "     - Usage: Can use pre-extracted latents, but no raw images"
echo ""
echo "   Difference:"
DIFF=$(echo "scale=2; 12 - $JENS_SIZE_GB" | bc)
echo "     Your dataset is ${DIFF} GB larger"
echo "     (Because it contains raw images, not just processed results)"

echo ""
echo "4. Recommendation:"
echo ""
echo "   ✅ Keep your dataset (12 GB)"
echo "      - You have the raw images needed for your analysis"
echo "      - Jens only has processed results"
echo "      - 12 GB is small compared to available staging space (2.0P)"
echo ""
echo "   💡 If you want to save space later:"
echo "      - After extracting latents, you could remove extracted directory"
echo "      - Keep only tar.gz (12 GB) for re-extraction when needed"
echo "      - But 12 GB is negligible, so not necessary"

