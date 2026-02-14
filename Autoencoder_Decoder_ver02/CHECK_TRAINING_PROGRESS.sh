#!/bin/bash
# Check why training hasn't started yet

JOB_ID="2650361.0"

echo "=== Checking Training Progress ==="
echo ""

# 1. Check task status
echo "1. Task Status:"
condor_q $JOB_ID
echo ""

# 2. Check latest output
echo "2. Latest Output (last 30 lines):"
if [ -f "logs/train_${JOB_ID//./_}_0.out" ]; then
    tail -n 30 "logs/train_${JOB_ID//./_}_0.out"
else
    echo "Output log not found yet"
fi
echo ""

# 3. Check for errors
echo "3. Errors (if any):"
if [ -f "logs/train_${JOB_ID//./_}_0.err" ]; then
    ERR_SIZE=$(stat -f%z "logs/train_${JOB_ID//./_}_0.err" 2>/dev/null || stat -c%s "logs/train_${JOB_ID//./_}_0.err" 2>/dev/null || echo "0")
    if [ "$ERR_SIZE" != "0" ]; then
        cat "logs/train_${JOB_ID//./_}_0.err"
    else
        echo "No errors (good!)"
    fi
else
    echo "Error log not found"
fi
echo ""

# 4. Check what stage it's at
echo "4. Current Stage:"
if [ -f "logs/train_${JOB_ID//./_}_0.out" ]; then
    OUTPUT=$(cat "logs/train_${JOB_ID//./_}_0.out")
    
    if echo "$OUTPUT" | grep -qi "epoch\|training\|starting training"; then
        echo "✅ Training has started!"
        echo "$OUTPUT" | grep -i "epoch\|training" | tail -5
    elif echo "$OUTPUT" | grep -qi "loading dataset\|dataset size"; then
        echo "⏳ Loading dataset..."
    elif echo "$OUTPUT" | grep -qi "installing\|downloading\|building wheel"; then
        echo "⏳ Still installing dependencies..."
    elif echo "$OUTPUT" | grep -qi "building index\|index.csv"; then
        echo "⏳ Building index.csv..."
    else
        echo "⏳ Initializing (check output above)"
    fi
else
    echo "⏳ Output log not created yet"
fi
echo ""

# 5. Check how long it's been running
echo "5. Runtime:"
condor_q -long $JOB_ID | grep -i "RemoteWallClockTime\|TotalJobRunTime" | head -2
echo ""

echo "=== What to Expect ==="
echo "Normal stages:"
echo "1. Installing dependencies (2-5 minutes)"
echo "2. Building index.csv (1-2 minutes)"
echo "3. Loading dataset (1-2 minutes)"
echo "4. Initializing model (10-30 seconds)"
echo "5. Starting training (then you'll see epochs)"
echo ""
echo "If stuck at dependency installation, that's normal - it takes time!"





