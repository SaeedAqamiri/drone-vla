#!/usr/bin/env python3
"""Restyle a run's satellite reference crops with FLUX.3-Klein.

Creates a sibling run directory that ``sat2aerial.cli localize`` can consume
unchanged: ``meta.json`` and ``poses.csv`` are copied, the aerial queries are
copied, and every ``pairs/satellite/<stem>.jpg`` is replaced by a FLUX restyle
of the same crop.

The crop is padded (symmetric, edge-replicate) up to a multiple of 16 so that
pixel coordinates -- and therefore the georeference in ``meta.json`` -- stay
valid; no rescaling is applied.

Example::

    python scripts/flux_augment_refs.py --run data/uav_visloc/processed_v1/06 \
        --out data/uav_visloc/processed_v1/06_flux --limit 6 --stride 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]

GEOMETRY_PROMPT = (
    "aerial photograph of exactly this scene taken by a small fixed-wing drone camera "
    "at nadir, keep every building, road, field boundary and tree in exactly the same "
    "place and the same size, only change the capture style: real-world daylight, "
    "slight atmospheric haze, soft lens, mild sensor noise, no text, no watermark"
)


def pad_to_block(image: Image.Image, block: int = 16) -> tuple[Image.Image, tuple[int, int]]:
    """Symmetric edge padding so width/height are multiples of ``block``."""
    width, height = image.size
    pad_w = (-width) % block
    pad_h = (-height) % block
    if not pad_w and not pad_h:
        return image, (0, 0)
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = Image.new(image.mode, (width + pad_w, height + pad_h))
    # fill with edge pixels: keeps the border plausible and lets the model see a
    # continuous scene instead of a hard frame
    padded.paste(image, (left, top))
    if left:
        padded.paste(image.crop((0, 0, 1, height)).resize((left, height)), (0, top))
    if right:
        padded.paste(image.crop((width - 1, 0, width, height)).resize((right, height)),
                     (left + width, top))
    if top:
        padded.paste(padded.crop((0, top, width + pad_w, top + 1)).resize((width + pad_w, top)),
                     (0, 0))
    if bottom:
        padded.paste(padded.crop((0, top + height - 1, width + pad_w, top + height)).resize(
            (width + pad_w, bottom)), (0, top + height))
    return padded, (left, top)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default=GEOMETRY_PROMPT)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--gateway", default="http://127.0.0.1:8090")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    run = Path(args.run)
    out = Path(args.out)
    sources = sorted((run / "pairs" / "satellite").glob("*.jpg"))
    if not sources:
        raise SystemExit(f"no reference crops under {run / 'pairs' / 'satellite'}")
    if args.stride > 1:
        sources = sources[:: args.stride]
    if args.limit:
        sources = sources[: args.limit]

    (out / "pairs" / "satellite").mkdir(parents=True, exist_ok=True)
    (out / "pairs" / "aerial").mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "meta.json", out / "meta.json")
    shutil.copy2(run / "poses.csv", out / "poses.csv")

    records = []
    for index, source in enumerate(sources, start=1):
        target = out / "pairs" / "satellite" / source.name
        aerial = run / "pairs" / "aerial" / f"{source.stem}_00.jpg"
        if aerial.is_file():
            shutil.copy2(aerial, out / "pairs" / "aerial" / aerial.name)
        if target.is_file() and not args.overwrite:
            print(f"[{index}/{len(sources)}] {source.stem}: cached")
            continue
        with Image.open(source) as raw:
            padded, origin = pad_to_block(raw.convert("RGB"))
        staged = out / f"_src_{source.stem}.png"
        padded.save(staged)
        command = [
            sys.executable, str(REPO / "scripts" / "flux_augment.py"),
            "--image", str(staged), "--out", str(target.with_suffix(".png")),
            "--prompt", args.prompt, "--steps", str(args.steps),
            "--seed", str(args.seed), "--gateway", args.gateway,
            "--width", str(padded.width), "--height", str(padded.height),
        ]
        print(f"[{index}/{len(sources)}] {source.stem}: {padded.width}x{padded.height}^{origin}")
        subprocess.run(command, check=True)
        generated = target.with_suffix(".png")
        with Image.open(generated) as image:
            image.convert("RGB").save(target, quality=95)
        generated.unlink()
        staged.unlink()
        records.append({"frame": source.stem, "size": [padded.width, padded.height],
                        "origin": list(origin), "seed": args.seed})

    (out / "flux_meta.json").write_text(
        json.dumps({"prompt": args.prompt, "steps": args.steps, "frames": records}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
