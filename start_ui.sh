#!/usr/bin/env bash
# 在 tmux 裡啟動 RD-Agent log 儀表板 (streamlit)，detached 執行、易於管理。
#
# 用法:
#   bash start_ui.sh              # 啟動 (預設 port 19899, log-dir log/)
#   UI_PORT=8501 bash start_ui.sh # 覆寫 port
#   FORCE=1 bash start_ui.sh      # port 被非 tmux 程序占用時，仍強制在 tmux 另起
#
# 管理:
#   tmux attach -t ui             # 看畫面 (脫離: Ctrl-b 再按 d)
#   tmux kill-session -t ui       # 關閉 UI
set -euo pipefail

SESSION="${UI_SESSION:-ui}"
PORT="${UI_PORT:-19899}"
LOG_DIR="${UI_LOG_DIR:-log/}"
CONDA_ENV="${UI_CONDA_ENV:-rdagent}"
CONDA_SH="/home/intern2/miniconda3/etc/profile.d/conda.sh"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) 已有同名 tmux session → 不重複啟動
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "✓ tmux session '$SESSION' 已存在，UI 應該已在跑。"
  echo "  看畫面: tmux attach -t $SESSION   (脫離: Ctrl-b d)"
  echo "  連線:   http://<host-or-tailscale-ip>:$PORT"
  exit 0
fi

# 2) port 被別的(非 tmux)程序占用 → 預設擋下，避免衝突 (FORCE=1 可略過)
if ss -ltn 2>/dev/null | grep -q ":$PORT\b"; then
  if [[ "${FORCE:-0}" != "1" ]]; then
    echo "✗ port $PORT 已被占用(可能是先前手動起的 UI,不在 tmux 裡)。"
    echo "  先停掉它,或用不同 port:  UI_PORT=8501 bash start_ui.sh"
    echo "  確定要在 tmux 另起(會與現有 UI 搶 port):  FORCE=1 bash start_ui.sh"
    exit 1
  fi
  echo "⚠ port $PORT 已被占用,FORCE=1 仍繼續(streamlit 可能改用相鄰 port)。"
fi

# 3) 在 detached tmux session 裡啟動
tmux new-session -d -s "$SESSION" -c "$REPO" \
  "source '$CONDA_SH' && conda activate '$CONDA_ENV' && exec rdagent ui --port '$PORT' --log-dir '$LOG_DIR'"

echo "✓ 已在 tmux session '$SESSION' 啟動 rdagent UI"
echo "  port=$PORT  log-dir=$LOG_DIR  env=$CONDA_ENV  cwd=$REPO"
echo "  看畫面: tmux attach -t $SESSION   (脫離: Ctrl-b d)"
echo "  連線:   http://<host-or-tailscale-ip>:$PORT"
