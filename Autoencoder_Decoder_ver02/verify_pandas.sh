#!/bin/bash
# 驗證腳本：檢查 run_train.sh 中是否包含 pandas 安裝和驗證

echo "=== 檢查 run_train.sh 中的 pandas 配置 ==="
echo ""

# 1. 檢查是否安裝 pandas
echo "1. 檢查 pandas 安裝命令："
if grep -q "pip install.*pandas" run_train.sh; then
    echo "   ✓ 找到 pandas 安裝命令"
    grep "pip install.*pandas" run_train.sh | head -3
else
    echo "   ✗ ERROR: 未找到 pandas 安裝命令"
    exit 1
fi

echo ""

# 2. 檢查 pandas 驗證
echo "2. 檢查 pandas 驗證："
PANDAS_CHECKS=$(grep -c "import pandas" run_train.sh)
if [ "$PANDAS_CHECKS" -gt 0 ]; then
    echo "   ✓ 找到 $PANDAS_CHECKS 處 pandas 驗證"
    grep -n "import pandas" run_train.sh
else
    echo "   ✗ ERROR: 未找到 pandas 驗證"
    exit 1
fi

echo ""

# 3. 檢查 venv 創建和激活
echo "3. 檢查 venv 創建和激活："
if grep -q "python3 -m venv" run_train.sh; then
    echo "   ✓ 找到 venv 創建命令"
else
    echo "   ✗ ERROR: 未找到 venv 創建命令"
    exit 1
fi

if grep -q "source .venv/bin/activate" run_train.sh; then
    VENV_ACTIVATIONS=$(grep -c "source .venv/bin/activate" run_train.sh)
    echo "   ✓ 找到 $VENV_ACTIVATIONS 處 venv 激活"
else
    echo "   ✗ ERROR: 未找到 venv 激活命令"
    exit 1
fi

echo ""

# 4. 檢查是否使用 venv 中的 Python
echo "4. 檢查是否使用 venv 中的 Python："
if grep -q 'PYTHON_BIN="\${VIRTUAL_ENV}/bin/python"' run_train.sh; then
    echo "   ✓ 使用 venv 中的 Python 絕對路徑"
else
    echo "   ⚠ WARNING: 未明確使用 venv 中的 Python 絕對路徑"
fi

echo ""

# 5. 總結
echo "=== 總結 ==="
echo "✓ pandas 安裝命令：存在"
echo "✓ pandas 驗證：$PANDAS_CHECKS 處"
echo "✓ venv 創建：存在"
echo "✓ venv 激活：$VENV_ACTIVATIONS 處"
echo ""
echo "✅ run_train.sh 包含完整的 pandas 配置"
echo ""

