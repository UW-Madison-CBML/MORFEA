#!/bin/bash
# Check job status using local logs (no staging access needed)

echo "=== Job Status Check (Local Logs Only) ==="
echo ""

# 1. Job history
echo "1. Recent Job History:"
condor_history | head -5
echo ""

# 2. Latest log files
echo "2. Latest Log Files:"
ls -lht logs/ 2>/dev/null | head -5 || echo "No logs directory found"
echo ""

# 3. Most recent output
echo "3. Most Recent Output Log:"
LATEST_OUT=$(ls -t logs/train_*.out 2>/dev/null | head -1)
if [ -n "$LATEST_OUT" ]; then
    echo "File: $LATEST_OUT"
    echo "--- Last 50 lines ---"
    tail -n 50 "$LATEST_OUT"
else
    echo "No output logs found"
fi
echo ""

# 4. Most recent errors
echo "4. Most Recent Error Log:"
LATEST_ERR=$(ls -t logs/train_*.err 2>/dev/null | head -1)
if [ -n "$LATEST_ERR" ]; then
    ERR_SIZE=$(stat -f%z "$LATEST_ERR" 2>/dev/null || stat -c%s "$LATEST_ERR" 2>/dev/null || echo "0")
    if [ "$ERR_SIZE" != "0" ]; then
        echo "File: $LATEST_ERR"
        echo "--- Error content ---"
        cat "$LATEST_ERR"
    else
        echo "Error log is empty (good!)"
    fi
else
    echo "No error logs found"
fi
echo ""

# 5. Training evidence
echo "5. Training Evidence:"
if grep -qi "epoch\|loss\|training\|starting" logs/train_*.out 2>/dev/null; then
    echo "✅ Training output found:"
    grep -i "epoch\|loss\|Epoch" logs/train_*.out | tail -10
else
    echo "❌ No training output found"
fi
echo ""

# 6. Checkpoints
echo "6. Local Checkpoints:"
if [ -d "checkpoints" ]; then
    ls -lht checkpoints/ | head -5
    echo "✅ Checkpoints directory exists"
else
    echo "❌ No checkpoints directory"
fi
echo ""

# 7. HTCondor log
echo "7. HTCondor Log (last 30 lines):"
LATEST_LOG=$(ls -t logs/train_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    tail -n 30 "$LATEST_LOG"
else
    echo "No HTCondor log found"
fi
echo ""

# 8. Summary
echo "=== Summary ==="
if grep -qi "epoch" logs/train_*.out 2>/dev/null; then
    EPOCHS=$(grep -i "epoch" logs/train_*.out | grep -oE "Epoch [0-9]+" | tail -1)
    if [ -n "$EPOCHS" ]; then
        echo "✅ Training ran: $EPOCHS"
    else
        echo "⚠️  Training started but unclear progress"
    fi
else
    echo "❌ No evidence of training"
fi

if [ -d "checkpoints" ] && [ "$(ls -A checkpoints/ 2>/dev/null)" ]; then
    echo "✅ Checkpoints found locally"
else
    echo "❌ No checkpoints found"
fi

echo ""
echo "Note: Results may be in staging (requires group access)"
echo "Contact lab admin or check with: ls -lh /staging/groups/bhaskar_group/ivf/results/"





