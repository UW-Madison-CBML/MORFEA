#!/bin/bash

echo ""

LOG_PATHS=(
    "/staging/groups/bhaskar_group/rho9/tphate_run.log"
    "$HOME/tphate_run.log"
    "./tphate_run.log"
)

for log_path in "${LOG_PATHS[@]}"; do
    if [ -f "$log_path" ]; then
        tail -20 "$log_path"
        echo ""
    fi
done

PID=$(ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
fi

echo ""
echo ""
echo "  watch -n 10 'ls -1 aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l'"






