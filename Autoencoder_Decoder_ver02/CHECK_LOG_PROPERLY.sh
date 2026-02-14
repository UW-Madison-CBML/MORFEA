#!/bin/bash
# 正确检查日志文件

echo "=== 检查日志文件位置 ==="
echo "当前目录: $(pwd)"
echo ""

# 检查可能的日志位置
LOG_PATHS=(
    "/staging/groups/bhaskar_group/rho9/tphate_run.log"
    "$HOME/tphate_run.log"
    "./tphate_run.log"
)

for log_path in "${LOG_PATHS[@]}"; do
    if [ -f "$log_path" ]; then
        echo "✓ 找到日志: $log_path"
        echo "  文件大小: $(ls -lh "$log_path" | awk '{print $5}')"
        echo "  最后 20 行:"
        tail -20 "$log_path"
        echo ""
    fi
done

echo "=== 检查进程输出 ==="
# 如果日志文件不存在，可能是输出还没写入
# 检查进程的 stdout/stderr
PID=$(ps aux | grep "generate_tphate_for_aadhitya.py" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "进程 PID: $PID"
    echo "检查进程的文件描述符:"
    ls -l /proc/$PID/fd/ 2>/dev/null | grep -E "(1|2)" || echo "无法访问 /proc/$PID/fd/"
fi

echo ""
echo "=== 建议 ==="
echo "如果日志文件不存在或为空，可能是："
echo "  1. 脚本还在初始化（加载数据）"
echo "  2. 输出被重定向到其他地方"
echo ""
echo "可以等待几分钟后再检查，或者直接监控 plot 数量："
echo "  watch -n 10 'ls -1 aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l'"






