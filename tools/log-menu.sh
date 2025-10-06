#!/usr/bin/env bash
set -euo pipefail

# Configurable dirs (override with env if you like)
LOGS_DIR="${LOGS_DIR:-logs}"
OUT_DIR="${OUT_DIR:-outputs}"
PAGER_CMD=${PAGER:-less -R}   # fallback to cat if less not installed

mkdir -p "$LOGS_DIR"

# ----- helpers -----
latest_with_ext() {  # $1 = ext (out|err). prints path or empty
  local ext="$1"
  local f=''
  if command -v find >/dev/null 2>&1; then
    # GNU find path (HPCs usually have this)
    f=$(find "$LOGS_DIR" -type f -name "*.${ext}" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -n1 | cut -d' ' -f2- || true)
  fi
  if [[ -z "${f:-}" ]]; then
    # Fallback if -printf isn't supported
    f=$(ls -t "$LOGS_DIR"/*."${ext}" 2>/dev/null | head -n1 || true)
  fi
  [[ -n "${f:-}" ]] && echo "$f" || return 1
}

latest_run_dir() {   # prints latest outputs/<run> dir (no trailing slash)
  local d=''
  d=$(ls -td "${OUT_DIR}"/*/ 2>/dev/null | head -n1 || true)
  [[ -n "${d:-}" ]] && printf '%s\n' "${d%/}" || return 1
}

pause() { read -rp $'\nPress Enter to continue...'; }

view_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    if command -v ${PAGER_CMD%% *} >/dev/null 2>&1; then
      ${PAGER_CMD} "$f" || cat "$f"
    else
      cat "$f"
    fi
  else
    echo "File not found: $f"
  fi
}

browse_outputs() {
  local dir
  if ! dir=$(latest_run_dir); then
    echo "No outputs/ runs found."
    return 1
  fi
  echo "Latest outputs dir: $dir"
  if command -v fzf >/dev/null 2>&1; then
    local target
    target=$(find "$dir" -type f 2>/dev/null | fzf --prompt="Pick a file > ") || return 1
    view_file "$target"
  else
    echo "Files (depth ≤2):"
    find "$dir" -maxdepth 2 -type f 2>/dev/null | nl -w2 -s') '
    read -rp "Enter relative path inside outputs (default .hydra/config.yaml): " rel
    rel=${rel:-.hydra/config.yaml}
    view_file "$dir/$rel"
  fi
}

# ----- menu -----
while :; do
  cat <<'EOF'

======== CULI Log/Run Menu ========
1) View latest logs/*.out
2) View latest logs/*.err
3) Browse latest outputs/<run>/... (open a file)
4) tail -f latest logs/*.out
5) Quit
EOF
  read -rp "Choose [1-5]: " choice
  case "$choice" in
    1)
      if f=$(latest_with_ext out); then view_file "$f"; else echo "No .out logs."; fi
      pause
      ;;
    2)
      if f=$(latest_with_ext err); then view_file "$f"; else echo "No .err logs."; fi
      pause
      ;;
    3)
      browse_outputs || true
      pause
      ;;
    4)
      if f=$(latest_with_ext out); then
        echo "Following: $f (Ctrl-C to stop)"
        tail -n 200 -f "$f"
      else
        echo "No .out logs."
        pause
      fi
      ;;
    5) exit 0 ;;
    *) echo "Invalid choice." ;;
  esac
done
