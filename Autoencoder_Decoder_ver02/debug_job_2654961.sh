#!/bin/bash
# Debug job 2654961

JOB_ID="2654961.0"
JOB_NUM="2654961"

echo "=== 1. Job History (檢查任務是否完成/失敗) ==="
condor_history $JOB_NUM -limit 1

echo ""
echo "=== 2. Job Details (退出代碼、狀態等) ==="
condor_history $JOB_NUM -limit 1 -long | grep -E "JobStatus|ExitCode|HoldReason|RemoteHost|RemoteWallClockTime" | head -10

echo ""
echo "=== 3. 檢查日誌文件是否存在 ==="
cd ~/ivf/Raffael/2025-11-19 2>/dev/null || cd ~/ivf_train 2>/dev/null || pwd
ls -lh logs/train_${JOB_NUM}_0.* 2>/dev/null || echo "❌ 沒有找到日誌文件"

echo ""
echo "=== 4. Error Log (錯誤日誌) ==="
if [ -f "logs/train_${JOB_NUM}_0.err" ]; then
    ERR_SIZE=$(stat -f%z "logs/train_${JOB_NUM}_0.err" 2>/dev/null || stat -c%s "logs/train_${JOB_NUM}_0.err" 2>/dev/null || echo "0")
    if [ "$ERR_SIZE" != "0" ]; then
        echo "❌ 發現錯誤："
        cat "logs/train_${JOB_NUM}_0.err"
    else
        echo "✅ 沒有錯誤（文件為空）"
    fi
else
    echo "❌ 錯誤日誌文件不存在"
fi

echo ""
echo "=== 5. Output Log (輸出日誌 - 最後 100 行) ==="
if [ -f "logs/train_${JOB_NUM}_0.out" ]; then
    echo "輸出文件大小: $(ls -lh logs/train_${JOB_NUM}_0.out | awk '{print $5}')"
    echo ""
    echo "最後 100 行："
    tail -n 100 "logs/train_${JOB_NUM}_0.out"
else
    echo "❌ 輸出日誌文件不存在"
fi

echo ""
echo "=== 6. HTCondor Log (系統日誌) ==="
if [ -f "logs/train_${JOB_NUM}_0.log" ]; then
    echo "檢查關鍵錯誤："
    grep -i "error\|failed\|exception\|traceback\|file.*not found\|module.*not found" "logs/train_${JOB_NUM}_0.log" | tail -20 || echo "沒有發現明顯錯誤"
    echo ""
    echo "最後 50 行："
    tail -n 50 "logs/train_${JOB_NUM}_0.log"
else
    echo "❌ HTCondor 日誌文件不存在"
fi

echo ""
echo "=== 7. Python 錯誤追蹤 (如果有的話) ==="
if [ -f "logs/train_${JOB_NUM}_0.out" ]; then
    grep -A 30 "Traceback\|Error\|Exception\|ModuleNotFoundError\|FileNotFoundError" "logs/train_${JOB_NUM}_0.out" | tail -50 || echo "沒有發現 Python 錯誤"
fi

echo ""
echo "=== 8. 檢查關鍵文件是否存在 ==="
echo "檢查 model.py 是否被傳輸："
if [ -f "logs/train_${JOB_NUM}_0.out" ]; then
    grep -i "model.py\|transfer\|file.*not found" "logs/train_${JOB_NUM}_0.out" | tail -10 || echo "沒有相關信息"
fi

echo ""
echo "=== 9. 檢查結果文件 ==="
if [ -d "checkpoints" ]; then
    echo "✅ checkpoints 目錄存在："
    ls -lh checkpoints/ | head -10
else
    echo "❌ checkpoints 目錄不存在"
fi

if [ -f "logs/training_log.json" ]; then
    echo "✅ training_log.json 存在"
    tail -n 20 logs/training_log.json
else
    echo "❌ training_log.json 不存在"
fi

echo ""
echo "=== 總結 ==="
echo "如果任務失敗，常見原因："
echo "1. model.py 文件缺失"
echo "2. 依賴包安裝失敗"
echo "3. 數據路徑問題"
echo "4. GPU 不可用"
echo ""
echo "查看完整輸出：cat logs/train_${JOB_NUM}_0.out"
echo "查看完整錯誤：cat logs/train_${JOB_NUM}_0.err"

