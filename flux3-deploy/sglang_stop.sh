#!/usr/bin/env bash
set -uo pipefail
ROOT="${FLUX3_ROOT:-/home/ava/flux3-klein}"
for f in "$ROOT"/logs/sglang_*.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f" 2>/dev/null || true)
  [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null || true
  rm -f "$f"
done
sleep 3
pkill -f "$ROOT/venv-sglang/bin/sglang serve" 2>/dev/null || true
pkill -f "$ROOT/sglang-weights" 2>/dev/null || true
sleep 1
pkill -9 -f "$ROOT/venv-sglang/bin/sglang serve" 2>/dev/null || true
echo "stopped"
