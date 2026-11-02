#!/usr/bin/env bash
# Re-apply the runtime patches needed by this offline FLUX.3/SGLang install.
# JIT kernels crash with "invalid resource handle" in the spawned scheduler on
# this CUDA toolchain, so the affected paths use their PyTorch fallbacks.
set -euo pipefail
ROOT="${FLUX3_ROOT:-/home/ava/flux3-klein}"
SITE="$ROOT/sglang-site"
python3 - "$SITE" <<'PY'
import pathlib, sys
site = pathlib.Path(sys.argv[1])

p = site / "sglang/srt/layers/quantization/fp8_utils.py"
s = p.read_text()
if "if False:\n                qinput, x_scale = sglang_per_token_quant_fp8(input_2d)" not in s:
    old = "            if _is_cuda:\n                qinput, x_scale = sglang_per_token_quant_fp8(input_2d)\n            else:\n"
    assert old in s
    p.write_text(s.replace(old, "            if False:\n                qinput, x_scale = sglang_per_token_quant_fp8(input_2d)\n            else:\n", 1))

p = site / "sglang/kernels/ops/quantization/__init__.py"
s = p.read_text()
if "scale_inv = torch.where(scale == 0" not in s:
    old = '''    """Per-token FP8 quantization, writing into ``output_q`` / ``output_s``."""
    return get_kernel("quantization.sgl_per_token_quant_fp8", KernelBackend.JIT)(
        input, output_q, output_s
    )'''
    new = '''    """Per-token FP8 quantization, writing into ``output_q`` / ``output_s``."""
    import torch
    amax = input.abs().amax(dim=1, keepdim=True)
    scale = amax / 448.0
    scale_inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    output_s.copy_(scale)
    output_q.copy_((input * scale_inv).clamp(-448.0, 448.0).to(output_q.dtype))'''
    assert old in s
    p.write_text(s.replace(old, new, 1))

p = site / "sglang/multimodal_gen/runtime/layers/rotary_embedding/utils.py"
s = p.read_text()
if "use_flashinfer = False" not in s:
    old = """    use_flashinfer = (
        flashinfer_apply_rope_inplace is not None
        and q.is_cuda
        and k.is_cuda
        and cos_sin_cache.is_cuda
        and q_heads == k_heads
    )
"""
    assert old in s
    p.write_text(s.replace(old, "    use_flashinfer = False\n", 1))
print("patches ok")
PY
SP="$ROOT/venv-sglang/lib/python3.12/site-packages"
printf '%s\n' "$SITE" 'import sys, os, nvidia_cutlass_dsl; sys.path.insert(0, os.path.join(nvidia_cutlass_dsl.__path__[0], "dsl_packages"))' > "$SP/sglang-site.pth"
rm -f "$SP/distutils-precedence.pth" "$SP/nvidia_cutlass_dsl_packages.pth"
echo "pth ok"
