#!/bin/bash

echo ""

if grep -q "pip install.*pandas" run_train.sh; then
    grep "pip install.*pandas" run_train.sh | head -3
else
    exit 1
fi

echo ""

PANDAS_CHECKS=$(grep -c "import pandas" run_train.sh)
if [ "$PANDAS_CHECKS" -gt 0 ]; then
    grep -n "import pandas" run_train.sh
else
    exit 1
fi

echo ""

if grep -q "python3 -m venv" run_train.sh; then
else
    exit 1
fi

if grep -q "source .venv/bin/activate" run_train.sh; then
    VENV_ACTIVATIONS=$(grep -c "source .venv/bin/activate" run_train.sh)
else
    exit 1
fi

echo ""

if grep -q 'PYTHON_BIN="\${VIRTUAL_ENV}/bin/python"' run_train.sh; then
else
fi

echo ""

echo ""
echo ""

