#!/bin/bash
# 檢查當前任務狀態

echo "============================================================"
echo "檢查當前任務狀態"
echo "============================================================"
echo ""
echo "請在 CHTC 上執行："
echo ""
echo "1. 檢查任務是否還在運行："
echo "   condor_q -submitter rho9 | grep extract"
echo ""
echo "2. 查看完整的輸出日誌（看 DEVICE 實際值）："
echo "   cat ~/logs/extract_latents_v1_baseline.out | grep -A 5 -B 5 'Device:'"
echo ""
echo "3. 查看任務是否還在運行或已完成："
echo "   condor_q -submitter rho9"
echo "   condor_history -limit 1 -submitter rho9 | grep extract"
echo ""
echo "4. 如果任務已完成，查看退出代碼："
echo "   condor_history -limit 1 2853038 -long | grep -E 'ExitCode|ExitStatus'"
echo ""
echo "============================================================"
echo "如果 DEVICE 變量仍然是空的："
echo "  - 可能是 Python 命令失敗但錯誤被隱藏了"
echo "  - 或者變量設置有問題"
echo "  - 但修復代碼應該會設置默認值為 'cuda'"
echo "============================================================"








