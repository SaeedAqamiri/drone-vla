#!/usr/bin/env bash
set -uo pipefail
CALLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$CALLER_DIR/comfy_stop.sh" || true
bash "$CALLER_DIR/sglang_stop.sh" || true
