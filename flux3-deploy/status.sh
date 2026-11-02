#!/usr/bin/env bash
set -uo pipefail
ROOT="${FLUX3_ROOT:-/home/ava/flux3-klein}"
GW="${GATEWAY_PORT:-8090}"
SG="${SGLANG_PORT:-30010}"
echo "== gateway =="; curl -s -m 3 "http://127.0.0.1:$GW/v1/health" || echo down
echo; echo "== queue =="; curl -s -m 3 "http://127.0.0.1:$GW/v1/queue" || echo down
echo; echo "== sglang =="; curl -s -m 3 "http://127.0.0.1:$SG/v1/models" || echo down
echo; echo "== gpu =="; nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "== procs =="; pgrep -af "$ROOT/ComfyUI/main.py|$ROOT/api/comfy_gateway.py|$ROOT/venv-sglang/bin/sglang serve" | sed "s#$ROOT/##" || true
