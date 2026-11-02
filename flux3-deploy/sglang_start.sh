#!/usr/bin/env bash
set -euo pipefail

ROOT="${FLUX3_ROOT:-/home/ava/flux3-klein}"
BIN="${SGLANG_BIN:-$ROOT/venv-sglang/bin/sglang}"
MODEL="${SGLANG_MODEL_PATH:-$ROOT/sglang-weights}"
PORT="${SGLANG_PORT:-30010}"
BATCH="${SGLANG_BATCH_SIZE:-2}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$ROOT/logs" "$ROOT/.hf-cache"
export CUDA_HOME="${CUDA_HOME:-$ROOT/sglang-site/nvidia/cu13}"
export PATH="$ROOT/venv-sglang/bin:$CUDA_HOME/bin:$PATH"

if [ -f "$ROOT/logs/sglang_$PORT.pid" ]; then
  old=$(cat "$ROOT/logs/sglang_$PORT.pid" 2>/dev/null || true)
  [ -n "${old:-}" ] && kill "$old" 2>/dev/null || true
  sleep 2
  rm -f "$ROOT/logs/sglang_$PORT.pid"
fi

echo "starting SGLang FLUX.3 Klein (fp8r) on 0.0.0.0:$PORT, batch=$BATCH"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME="$ROOT/.hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
SGLANG_FORCE_FUSED_OP_BACKEND="${SGLANG_FORCE_FUSED_OP_BACKEND:-torch}" \
SGLANG_ENABLE_FUSED_QKNORM_ROPE="${SGLANG_ENABLE_FUSED_QKNORM_ROPE:-0}" \
  nohup "$BIN" serve \
    --model-path "$MODEL" \
    --model-variant fp8r \
    --host 0.0.0.0 --port "$PORT" \
    --batching-max-size "$BATCH" \
    --dit-cpu-offload=false \
    --batching-delay-ms 5 \
    --log-level info \
    >"$ROOT/logs/sglang_$PORT.log" 2>&1 &
echo $! >"$ROOT/logs/sglang_$PORT.pid"

for _ in $(seq 1 600); do
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || \
     curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "SGLang ready: http://127.0.0.1:$PORT"
    exit 0
  fi
  sleep 2
done
echo "SGLang did not become ready; see $ROOT/logs/sglang_$PORT.log" >&2
exit 1
