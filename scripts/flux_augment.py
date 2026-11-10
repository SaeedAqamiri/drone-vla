#!/usr/bin/env python3
"""Call the remote FLUX.3-Klein gateway to restyle a satellite crop as a drone view.

The gateway (``flux3-deploy/comfy_gateway.py``) accepts a JSON job with a
``references`` list of base64 images; reference 0 is resized to the generation
size and fed through ``VAEEncode -> ReferenceLatent``, so it conditions the
output on the *actual* crop instead of inventing a scene from text alone.

Geometry matters more than beauty here: a restyled reference is only usable as a
map if the ground features stay where they were. ``scripts/eval_flux_geometry.py``
measures that after the fact.

Example::

    python scripts/flux_augment.py --image outputs/flux_aug/patch_a.png \
        --out outputs/flux_aug/patch_a_style1.png \
        --prompt "aerial photo taken by a fixed-wing drone ..."
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

DEFAULT_PROMPT = (
    "aerial photograph taken straight down by a fixed-wing drone camera at 700 m altitude, "
    "oblique-free nadir view, natural daylight, slight atmospheric haze, camera lens softness, "
    "JPEG compression artifacts, farm fields and small roads, no text, no watermark"
)


def load_reference(path: Path, width: int, height: int) -> str:
    image = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch(url: str, timeout: float = 120.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def size_for(path: Path, long_side: int) -> tuple[int, int]:
    """Generation size, multiple of 16, keeping the crop's aspect ratio."""
    with Image.open(path) as image:
        width, height = image.size
    scale = long_side / max(width, height)
    out_w = max(64, int(round(width * scale / 16)) * 16)
    out_h = max(64, int(round(height * scale / 16)) * 16)
    return out_w, out_h


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", required=True, help="satellite crop to restyle")
    parser.add_argument("--out", required=True, help="output PNG")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative", default="")
    parser.add_argument("--gateway", default="http://127.0.0.1:8090")
    parser.add_argument("--long-side", type=int, default=1024)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra-reference", action="append", default=[],
                        help="additional reference image (repeatable)")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    source = Path(args.image)
    if not source.is_file():
        raise SystemExit(f"missing {source}")
    width, height = (
        (args.width, args.height) if args.width and args.height else size_for(source, args.long_side)
    )

    references = [load_reference(source, width, height)]
    for extra in args.extra_reference:
        references.append(load_reference(Path(extra), width, height))

    payload: dict = {
        "prompt": args.prompt,
        "negative": args.negative,
        "width": width,
        "height": height,
        "steps": args.steps,
        "seed": args.seed,
        "references": references,
        "filename_prefix": "flux_aug",
    }
    if args.alpha:
        payload["alpha"] = args.alpha

    started = time.monotonic()
    job = post_json(f"{args.gateway}/v1/generate?wait=true&timeout={int(args.timeout)}",
                    payload, timeout=args.timeout + 60)
    status = job.get("status")
    if status != "succeeded":
        raise SystemExit(f"job {job.get('id')} finished as {status}: {job.get('error')}")
    data = fetch(f"{args.gateway}/v1/jobs/{job['id']}/image")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"{source.name} -> {out} ({width}x{height}, {time.monotonic() - started:.1f}s, "
          f"job {job['id']}, seed {job.get('seed')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
