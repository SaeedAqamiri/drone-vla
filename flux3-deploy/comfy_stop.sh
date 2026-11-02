#!/usr/bin/env bash
set -uo pipefail
ROOT="${FLUX3_ROOT:-/home/ava/flux3-klein}"

for f in "$ROOT"/logs/comfy_*.pid "$ROOT"/logs/gateway.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f" 2>/dev/null || true)
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$f"
done

sleep 2
# Catch children that outlived their shell / stale pid files.
pkill -f "$ROOT/ComfyUI/main.py" 2>/dev/null || true
pkill -f "$ROOT/api/comfy_gateway.py" 2>/dev/null || true
sleep 1
pkill -9 -f "$ROOT/ComfyUI/main.py" 2>/dev/null || true
pkill -9 -f "$ROOT/api/comfy_gateway.py" 2>/dev/null || true
echo "stopped"
