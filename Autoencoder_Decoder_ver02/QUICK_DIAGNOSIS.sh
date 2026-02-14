#!/bin/bash
# 快速診斷 SSH 連接問題

echo "=== SSH 連接診斷 ==="
echo ""

echo "1. 測試基本網絡連接..."
ping -c 3 ap2001.chtc.wisc.edu 2>&1 | head -5

echo ""
echo "2. 測試 SSH 端口 (22)..."
nc -zv -w 5 ap2001.chtc.wisc.edu 22 2>&1 || echo "無法連接到端口 22"

echo ""
echo "3. 檢查 DNS 解析..."
nslookup ap2001.chtc.wisc.edu 2>&1 | head -5

echo ""
echo "=== 建議 ==="
echo "如果以上都失敗，可能是："
echo "  - 網絡防火牆阻止了連接"
echo "  - 需要 VPN 連接"
echo "  - CHTC 服務器暫時不可用"
echo ""
echo "代碼修改已完成，一旦連接恢復就可以上傳使用。"








