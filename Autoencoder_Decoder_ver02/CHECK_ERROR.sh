#!/bin/bash
# Check actual training error

echo "=== 1. Error Log (CRITICAL) ==="
cat logs/train_2650510_0.err 2>/dev/null || echo "No .err file found"
echo ""

echo "=== 2. Full Output Log (showing Python errors) ==="
cat logs/train_2650510_0.out 2>/dev/null | grep -A 20 "Starting training" || echo "No training start found"
echo ""

echo "=== 3. Python Traceback (if any) ==="
cat logs/train_2650510_0.out 2>/dev/null | grep -A 30 "Traceback\|Error\|Exception" || echo "No traceback found"
echo ""

echo "=== 4. Last 100 lines of output ==="
tail -n 100 logs/train_2650510_0.out 2>/dev/null || echo "No output file"
echo ""

echo "=== 5. Check if train.py was found ==="
cat logs/train_2650510_0.out 2>/dev/null | grep -i "train.py\|python3\|command not found" | tail -5





