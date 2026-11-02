#!/usr/bin/env bash
set -euo pipefail

ROOT="${FLUX3_ROOT:-/home/ava/flux3-klein}"
PY="${COMFY_PYTHON:-$ROOT/venv-comfy/bin/python}"
N="${COMFY_WORKERS:-2}"
BASE_PORT="${COMFY_BASE_PORT:-8188}"
GW_PORT="${GATEWAY_PORT:-8090}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$ROOT/logs" "$ROOT/comfy_input" "$ROOT/comfy_output" "$ROOT/comfy_temp" "$ROOT/outputs"

if [ -x "$ROOT/api/comfy_stop.sh" ]; then
  "$ROOT/api/comfy_stop.sh" >/dev/null 2>&1 || true
fi

URLS=""
for i in $(seq 0 $((N - 1))); do
  port=$((BASE_PORT + i))
  echo "starting ComfyUI worker $i on 127.0.0.1:$port"
  extra=()
  [ "${COMFY_DISABLE_SMART_MEMORY:-0}" = "1" ] && extra+=(--disable-smart-memory)
  [ -n "${COMFY_RESERVE_VRAM:-}" ] && extra+=(--reserve-vram "$COMFY_RESERVE_VRAM")
  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PY" "$ROOT/ComfyUI/main.py" \
      --listen 127.0.0.1 --port "$port" \
      --input-directory "$ROOT/comfy_input" \
      --output-directory "$ROOT/comfy_output" \
      --temp-directory "$ROOT/comfy_temp" \
      --disable-auto-launch --preview-method none \
      "${extra[@]}" \
      >"$ROOT/logs/comfy_$port.log" 2>&1 &
  echo $! >"$ROOT/logs/comfy_$port.pid"
  URLS="${URLS}http://127.0.0.1:$port,"
done
URLS="${URLS%,}"

for i in $(seq 0 $((N - 1))); do
  port=$((BASE_PORT + i))
  ok=0
  for _ in $(seq 1 240); do
    if curl -sf -m 2 "http://127.0.0.1:$port/system_stats" >/dev/null 2>&1; then ok=1; break; fi
    sleep 1
  done
  if [ "$ok" != 1 ]; then
    echo "ComfyUI worker on port $port did not become ready; see $ROOT/logs/comfy_$port.log" >&2
    exit 1
  fi
  echo "worker $port ready"
done

export COMFY_URLS="$URLS"
export COMFY_INPUT_DIR="$ROOT/comfy_input"
export OUTPUT_DIR="$ROOT/outputs"
export DB_PATH="$ROOT/outputs/jobs.db"
export PORT="$GW_PORT"
echo "starting gateway on 0.0.0.0:$GW_PORT with COMFY_URLS=$COMFY_URLS"
nohup "$PY" "$ROOT/api/comfy_gateway.py" >"$ROOT/logs/gateway.log" 2>&1 &
echo $! >"$ROOT/logs/gateway.pid"

for _ in $(seq 1 60); do
  if curl -sf -m 2 "http://127.0.0.1:$GW_PORT/v1/health" >/dev/null 2>&1; then
    echo "gateway ready: http://127.0.0.1:$GW_PORT/docs"
    exit 0
  fi
  sleep 1
done
echo "gateway did not become ready; see $ROOT/logs/gateway.log" >&2
exit 1
