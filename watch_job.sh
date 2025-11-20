#!/usr/bin/env bash
# 用法：
#   ./watch_job.sh          # 追蹤最新一個 job
#   ./watch_job.sh 2651741  # 追蹤指定 job (ClusterId)

set -e

# 1) 取得 JOBID（不含 .0）
JOBID="$1"

if [ -z "$JOBID" ]; then
    echo "[info] 沒有給 job id，幫你抓最新一顆 job..."
    JOBID=$(condor_q "$USER" -autoformat ClusterId 2>/dev/null | tail -1)
fi

if [ -z "$JOBID" ]; then
    echo "[error] 找不到任何 job（condor_q 是空的）"
    exit 1
fi

echo "[info] 監看 job ${JOBID}.0"

# 2) 迴圈：只要 job 還在 queue 裡，就用 condor_tail 看 stdout
while true; do
    STATUS=$(condor_q "${JOBID}.0" -autoformat JobStatus 2>/dev/null || true)

    if [ -z "$STATUS" ]; then
        echo
        echo "[info] job ${JOBID}.0 已經不在 queue（可能完成或被移除）"
        break
    fi

    case "$STATUS" in
        1)
            echo
            echo "[info] job ${JOBID}.0 現在是 Idle（排隊中），等一下再看輸出..."
            ;;
        2|6)
            echo
            echo "========== [$(date)] condor_tail ${JOBID}.0 =========="
            # 顯示最近一段 stdout（最多 20KB）
            condor_tail -maxbytes 20000 "${JOBID}.0" || echo "[warn] condor_tail 失敗"
            ;;
        5)
            echo
            echo "[error] job ${JOBID}.0 被 Hold 了，細節如下："
            condor_q -hold "${JOBID}.0" || true
            break
            ;;
        *)
            echo
            echo "[info] job ${JOBID}.0 狀態碼 = ${STATUS}（非 Idle/Running），跳出迴圈"
            break
            ;;
    esac

    echo
    echo "[info] 60 秒後再更新（Ctrl-C 可隨時停止）..."
    sleep 60
done

# 3) job 離開 queue 後，嘗試顯示本地 logs 裡的 .out
echo
echo "[info] 嘗試顯示本地 logs 裡的最終 .out 檔："
LATEST_OUT=$(ls -t logs/train_"${JOBID}"_0.out 2>/dev/null | head -1 || true)
if [ -n "$LATEST_OUT" ]; then
    echo "[info] Latest output file: ${LATEST_OUT}"
    tail -n 80 "$LATEST_OUT"
else
    echo "[info] 還找不到 logs/train_${JOBID}_0.out（可能 job 被移除或還沒傳回來）"
fi
