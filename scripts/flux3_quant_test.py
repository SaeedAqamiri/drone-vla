"""FLUX 3 Action (7B) feasibility test on an 8GB GPU.

Builds the action DiT on the meta device, then streams the mmap'd base trunk
to the GPU tensor-by-tensor, quantizing nn.Linear weights on the fly
(nf4 / int8 via bitsandbytes). The KinoVAE goes to GPU in bf16 and the text
encoder is replaced by the repo's mock (same context dims), so host RAM stays
flat. Then runs one dummy predict_action_chunk at the drone/game geometry
(512x512 canvas, 4-dim actions, 1-step cosmos_unipc sampler).

Usage:
    python flux3_quant_test.py --quant nf4    # ~3.7GB trunk  -> should fit
    python flux3_quant_test.py --quant int8   # ~7GB trunk    -> borderline
    python flux3_quant_test.py --quant bf16   # ~14GB trunk   -> OOM expected
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/arman/drone-vla/flux-action/src")

import torch
from torch import nn
from safetensors import safe_open
import bitsandbytes as bnb

from flux_action.config import PolicyConfig, REQUIRED_CONTENT_STREAMS
from flux_action.models.text_encoder import load_text_encoder
from flux_action.models.transformer import JointSingleSeqParams
from flux_action.models.video_vae import load_video_vae
from flux_action.models.wiring import (
    OPTIONAL_KEY_SUBSTRINGS,
    action_dit_params,
    build_action_dit,
    fresh_head_state_dict,
    fresh_module_names,
    remap_shared_action_keys,
    stream_of_key,
    restrict_content_streams,
)
from flux_action.policy import FluxActionPolicy

MODELS = Path("/home/arman/drone-vla/models/flux-3-action-base")
TRUNK = MODELS / "flux-3-action-base.safetensors"
VAE = MODELS / "video_vae.safetensors"


def build_policy():
    policy_dict = json.loads(
        Path("/home/arman/drone-vla/flux-action/configs/games/train.json").read_text()
    )["policy"]
    policy_dict.update(
        trunk_weights=None,
        video_vae_id=None,
        text_encoder_id=None,
        compile_model=False,
    )
    cfg = PolicyConfig(**policy_dict)
    streams = cfg.content_streams or REQUIRED_CONTENT_STREAMS
    dit_params = action_dit_params(
        restrict_content_streams(JointSingleSeqParams(**cfg.dit_config), streams),
        cfg.action_modality,
        cfg.action_dim,
        attn_mode=cfg.attn_mode,
        conditioning_channels=cfg.conditioning_channels,
    )
    dit = build_action_dit(
        dit_params,
        None,
        modality=cfg.action_modality,
        device="meta",
        dtype=torch.bfloat16,
        head_seed=cfg.head_init_seed,
    )
    vae = load_video_vae(str(VAE), device="cuda")
    text_encoder = load_text_encoder(None)
    policy = FluxActionPolicy(cfg, video_vae=vae, text_encoder=text_encoder, _restore=True)
    policy.dit = dit
    return policy


def swap_linears(model, quant, protected_prefixes):
    linears = {}
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            full = f"{parent_name}.{child_name}".lstrip(".")
            if (
                isinstance(child, nn.Linear)
                and ".adaLN_modulation" not in full  # FinalLayer chunks the raw weight — needs bf16
                and not any(full == p.rstrip(".") or full.startswith(p) for p in protected_prefixes)
            ):
                linears[full] = (parent, child_name, child)
    for name, (parent, child_name, child) in linears.items():
        if quant == "nf4":
            new = bnb.nn.Linear4bit(
                child.in_features, child.out_features, bias=child.bias is not None,
                compute_dtype=torch.bfloat16, quant_type="nf4",
            )
        elif quant == "int8":
            new = bnb.nn.Linear8bitLt(
                child.in_features, child.out_features, bias=child.bias is not None,
                has_fp16_weights=False, threshold=6.0,
            )
        else:
            continue
        new.weight = nn.Parameter(
            torch.empty(child.out_features, child.in_features, dtype=torch.uint8, device="meta"),
            requires_grad=False,
        )
        if child.bias is not None:
            new.bias = nn.Parameter(
                torch.empty(child.out_features, dtype=torch.bfloat16, device="meta"),
                requires_grad=False,
            )
        setattr(parent, child_name, new)
    return set(linears)


def quant_weight(t, quant):
    if quant == "nf4":
        return bnb.nn.Params4bit(
            data=t, quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            quant_storage=torch.uint8,
        ).to("cuda")
    if quant == "int8":
        return bnb.nn.Int8Params(t.to("cuda"), requires_grad=False, has_fp16_weights=False)
    return t.to("cuda", torch.bfloat16)


def load_trunk(policy, quant):
    dit = policy.dit
    modality = policy.config.action_modality
    fresh_prefixes = tuple(n + "." for n in fresh_module_names(modality))
    linear_names = set()
    if quant in ("nf4", "int8"):
        linear_names = swap_linears(dit, quant, fresh_prefixes)
        print(f"swapped {len(linear_names)} Linear layers -> {quant}")

    unused = set(dit.in_channels.keys()) if hasattr(dit.in_channels, "keys") else set()
    import flux_action.models.wiring as wiring

    all_streams = set(wiring.CONTENT_STREAMS)
    unused = all_streams - set(dit.in_channels)

    assigned = set()
    chunk = {}
    with safe_open(TRUNK, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if stream_of_key(k) not in unused]
        for key, orig in remap_shared_action_keys({k: k for k in keys}, modality).items():
            t = f.get_tensor(orig)
            name = key
            if name.endswith(".weight") and name[: -len(".weight")] in linear_names:
                module = dict(dit.named_modules())[name[: -len(".weight")]]
                module.weight = quant_weight(t, quant)
            elif name.endswith(".bias") and name[: -len(".bias")] in linear_names and quant != "bf16":
                module = dict(dit.named_modules())[name[: -len(".bias")]]
                module.bias = nn.Parameter(t.to("cuda", torch.bfloat16))
            else:
                chunk[name] = t.to("cuda", torch.bfloat16)
            assigned.add(name)
            if len(chunk) >= 100:
                dit.load_state_dict(chunk, strict=False, assign=True)
                chunk = {}
    if chunk:
        dit.load_state_dict(chunk, strict=False, assign=True)

    def is_bnb_artifact(k):
        return (
            ".absmax" in k or "quant_map" in k or "quant_state" in k
            or ".nested" in k or k.endswith(".format")
        )

    missing = [
        k for k in dit.state_dict()
        if k not in assigned and not is_bnb_artifact(k)
        and not k.startswith(fresh_prefixes)
        and not any(s in k for s in OPTIONAL_KEY_SUBSTRINGS)
    ]
    if missing:
        raise ValueError(f"trunk missing required keys, e.g. {missing[:6]}")

    fresh_missing = [k for k in dit.state_dict() if k not in assigned and k.startswith(fresh_prefixes)]
    if fresh_missing:
        emb = dit.emb_in[modality].weight
        fresh = fresh_head_state_dict(
            emb.shape[0], modality, emb.shape[1], policy.config.head_init_seed,
            conditioning_channels=dit.emb_in[f"{modality}_cond"].in_features,
        )
        dit.load_state_dict(
            {k: fresh[k].to("cuda", torch.bfloat16) for k in fresh_missing}, strict=False, assign=True
        )
        print(f"initialized {len(fresh_missing)} fresh embodiment-head tensors")

    leftover = [n for n, p in dit.named_parameters() if p.device.type == "meta"]
    if leftover:
        raise RuntimeError(f"params still on meta: {leftover[:6]}")

    if quant in ("nf4", "int8"):
        import types
        from flux_action.models.transformer import timestep_embedding

        def embed_timesteps(self, ts, out_dtype):
            weight_dtype = torch.bfloat16  # quantized Params4bit stores uint8; keep MLP in bf16
            with torch.autocast(device_type=ts.device.type, enabled=False):
                vec = self.time_in(timestep_embedding(ts, 256, out_dtype=weight_dtype))
            return vec.to(out_dtype)

        policy.dit.embed_timesteps = types.MethodType(embed_timesteps, policy.dit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=["nf4", "int8", "bf16"], default="nf4")
    args = ap.parse_args()

    print(f"=== FLUX 3 Action feasibility | quant={args.quant} ===", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    policy = build_policy()
    print(f"policy wired ({time.time() - t0:.1f}s), trunk -> GPU ({args.quant}) ...", flush=True)
    load_trunk(policy, args.quant)
    policy.dtype_ = torch.bfloat16
    policy.reset()

    alloc = torch.cuda.memory_allocated() / 1e9
    print(f"GPU after load: {alloc:.2f} GB | load time {time.time() - t0:.1f}s", flush=True)

    obs = {
        "images.game": torch.rand(1, 3, 512, 512, device="cuda"),
        "state": torch.zeros(1, 4, device="cuda"),
        "task": ["fly the drone: take off and hover in place"],
    }
    with torch.inference_mode():
        t1 = time.time()
        plan = policy.predict_action_chunk(obs)
        first = time.time() - t1
        t2 = time.time()
        plan2 = policy.predict_action_chunk(obs)
        second = time.time() - t2
    peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"plan shape: {tuple(plan.shape)}  (expected (1, 32, 4))")
    print(f"plan time: first {first * 1000:.0f} ms | second {second * 1000:.0f} ms")
    print(f"peak GPU memory: {peak:.2f} GB")
    verdict = "FITS" if peak < 7.6 else "DOES NOT FIT"
    print(f"VERDICT on 8GB GPU: {verdict} ({args.quant})")


if __name__ == "__main__":
    main()
