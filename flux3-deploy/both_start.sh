#!/usr/bin/env bash
set -euo pipefail
ROOT="${FLUX3_ROOT:-/home/ava/flux3-klein}"
CALLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# SGLang holds the resident DiT; ComfyUI workers then load on demand and release
# VRAM between prompts so both engines can share one GPU.
"$CALLER_DIR/sglang_start.sh"
COMFY_WORKERS="${COMFY_WORKERS:-1}" COMFY_DISABLE_SMART_MEMORY=1 bash "$CALLER_DIR/comfy_start.sh"
