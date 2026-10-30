from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .backends import make_backend
from .bakeoff import BakeoffConfig, run_bakeoff
from .dataset import load_pairs
from .matching import match_images
from .metrics import evaluate_pair
from .prompts import build_prompt, parse_variants
from .sizing import SizingInput, compute as compute_sizing, format_report
from .synthetic import generate_demo, make_aerial, make_satellite


def parse_size(raw: str) -> tuple[int, int]:
    token = raw.lower().strip()
    if "x" in token:
        width, height = token.split("x", 1)
        return int(width), int(height)
    value = int(token)
    return value, value


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="dummy", choices=["dummy", "comfy", "bfl", "flux3"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--data-uri", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-id", default="black-forest-labs/flux-3-image-klein")
    parser.add_argument("--variant-name", default="fp8r")
    parser.add_argument("--no-offload", action="store_true")
    parser.add_argument("--perspective", type=float, default=0.035)
    parser.add_argument("--blur", type=float, default=0.8)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-input-dir", default="ComfyUI/input")
    parser.add_argument("--comfy-timeout", type=float, default=900.0)
    parser.add_argument("--comfy-alpha", type=float, default=16.0)


def add_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--view", default="nadir", choices=["nadir", "oblique15", "oblique30", "oblique45"])
    parser.add_argument("--altitude", type=int, default=120)
    parser.add_argument("--season", default="summer")
    parser.add_argument("--lighting", default="clear midday sunlight")
    parser.add_argument("--weather", default="clear")
    parser.add_argument("--prompt-extra", default="")
    parser.add_argument("--size", type=parse_size, default=(768, 768))


def backend_from_args(args):
    kwargs: dict = {"seed": args.seed}
    if args.backend == "bfl":
        kwargs.update(
            api_key=args.api_key,
            steps=args.steps,
            dry_run=args.dry_run,
            data_uri=args.data_uri,
        )
    elif args.backend == "flux3":
        kwargs.update(
            model_id=args.model_id,
            variant=args.variant_name,
            steps=args.steps,
            cpu_offload=not args.no_offload,
        )
    elif args.backend == "comfy":
        kwargs.update(
            server=args.comfy_url,
            input_dir=args.comfy_input_dir,
            steps=args.steps,
            alpha=args.comfy_alpha,
            timeout=args.comfy_timeout,
        )
    else:
        kwargs.update(perspective=args.perspective, blur=args.blur)
    return make_backend(args.backend, **kwargs)


def cmd_pairs(args: argparse.Namespace) -> int:
    pairs = load_pairs(args.pairs)
    for pair in pairs:
        print(f"{pair.site_id}\t{pair.satellite}\t{len(pair.aerials)} aerial(s)")
    print(f"total: {len(pairs)} pairs")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    backend = backend_from_args(args)
    style_refs = [Path(item) for item in args.style_ref]
    prompt = build_prompt(
        view=args.view,
        style_refs=len(style_refs),
        altitude_m=args.altitude,
        season=args.season,
        lighting=args.lighting,
        weather=args.weather,
        extra=args.prompt_extra,
    )
    meta = backend.generate(
        satellite=Path(args.satellite),
        prompt=prompt,
        out_path=Path(args.out),
        size=args.size,
        seed=args.seed,
        style_refs=style_refs,
    )
    print(json.dumps(meta, indent=2))
    return 0


def cmd_bakeoff(args: argparse.Namespace) -> int:
    pairs = load_pairs(args.pairs)
    backend = backend_from_args(args)
    config = BakeoffConfig(
        out_dir=Path(args.out),
        variants=parse_variants(args.variants),
        size=args.size,
        seed=args.seed,
        seeds=args.seeds,
        match_mode=args.match_mode,
        style_refs=args.style_refs,
        min_inliers=args.min_inliers,
        min_ratio=args.min_ratio,
        min_coverage=args.min_coverage,
        tolerance_px=args.tolerance_px,
        max_pairs=args.max_pairs,
        altitude_m=args.altitude,
        season=args.season,
        lighting=args.lighting,
        weather=args.weather,
        extra_prompt=args.prompt_extra,
        max_dim=args.max_dim,
    )
    result = run_bakeoff(pairs, backend, config)
    print(json.dumps(result.aggregate, indent=2))
    print(f"report: {result.out_dir / 'report.md'}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    data_root = generate_demo(out / "data", sites=args.sites, size=args.size[0], seed=args.seed)
    pairs = load_pairs(data_root)
    backend = backend_from_args(args)
    config = BakeoffConfig(
        out_dir=out / "bakeoff",
        variants=parse_variants(args.variants),
        size=args.size,
        seed=args.seed,
        seeds=args.seeds,
        match_mode=args.match_mode,
        style_refs=args.style_refs,
        min_inliers=args.min_inliers,
        min_ratio=args.min_ratio,
        min_coverage=args.min_coverage,
        max_pairs=args.max_pairs,
        altitude_m=args.altitude,
    )
    result = run_bakeoff(pairs, backend, config)
    print(json.dumps(result.aggregate, indent=2))
    print(f"demo data: {data_root}")
    print(f"report: {result.out_dir / 'report.md'}")
    return 0


def cmd_selfcheck(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    satellite = make_satellite(rng, size=512)
    aerial = make_aerial(satellite, rng)
    sat_path = out / "satellite.png"
    aerial_path = out / "aerial.png"
    cv2.imwrite(str(sat_path), satellite)
    cv2.imwrite(str(aerial_path), aerial)
    match, homography = match_images(sat_path, aerial_path, mode=args.match_mode)
    fidelity = evaluate_pair(aerial_path, sat_path, homography)
    payload = {"match": match.as_dict(), "fidelity": fidelity}
    print(json.dumps(payload, indent=2))
    if not match.success:
        print("selfcheck failed: synthetic pair did not match", file=sys.stderr)
        return 1
    return 0


def cmd_sizing(args: argparse.Namespace) -> int:
    result = compute_sizing(
        SizingInput(
            distance_km=args.distance_km,
            corridor_km=args.corridor_km,
            altitude_m=args.altitude_m,
            fov_deg=args.fov,
            speed_kmh=args.speed_kmh,
            camera_fps=args.camera_fps,
            keyframe_hz=args.keyframe_hz,
            match_every_m=args.match_every_m,
            satellite_gsd_m=args.satellite_gsd_m,
            drone_pixels_wide=args.drone_pixels_wide,
            tile_px=args.tile_px,
            gen_seconds=args.gen_seconds,
            api_seconds=args.api_seconds,
            api_cost_per_image=args.api_cost,
            gpu_count=args.gpus,
            match_seconds=args.match_seconds,
            frame_mb=args.frame_mb,
            generated_tile_mb=args.generated_tile_mb,
            compression_ratio=args.compression_ratio,
        )
    )
    print(format_report(result))
    if not args.quiet:
        print()
        print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sat2aerial",
        description="Satellite-to-aerial generation and map-matching bake-off.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    pairs = sub.add_parser("pairs", help="list discovered satellite/aerial pairs")
    pairs.add_argument("--pairs", required=True)
    pairs.set_defaults(func=cmd_pairs)

    generate = sub.add_parser("generate", help="generate one aerial reference from one satellite tile")
    generate.add_argument("--satellite", required=True)
    generate.add_argument("--out", required=True)
    generate.add_argument("--style-ref", action="append", default=[])
    add_backend_args(generate)
    add_prompt_args(generate)
    generate.set_defaults(func=cmd_generate)

    bakeoff = sub.add_parser("bakeoff", help="compare generation variants against a real drone baseline")
    bakeoff.add_argument("--pairs", required=True)
    bakeoff.add_argument("--out", required=True)
    bakeoff.add_argument("--variants", default="nadir,oblique30,nadir_style,oblique30_style")
    bakeoff.add_argument("--style-refs", type=int, default=1)
    bakeoff.add_argument("--seeds", type=int, default=1)
    bakeoff.add_argument("--match-mode", default="edges", choices=["edges", "gray"])
    bakeoff.add_argument("--min-inliers", type=int, default=15)
    bakeoff.add_argument("--min-ratio", type=float, default=0.15)
    bakeoff.add_argument("--min-coverage", type=float, default=0.03)
    bakeoff.add_argument("--tolerance-px", type=int, default=3)
    bakeoff.add_argument("--max-dim", type=int, default=1024)
    bakeoff.add_argument("--max-pairs", type=int, default=None)
    add_backend_args(bakeoff)
    add_prompt_args(bakeoff)
    bakeoff.set_defaults(func=cmd_bakeoff)

    demo = sub.add_parser("demo", help="offline end-to-end smoke test on procedural city scenes")
    demo.add_argument("--out", default="outputs/sat2aerial/demo")
    demo.add_argument("--sites", type=int, default=4)
    demo.add_argument("--variants", default="nadir,oblique30,nadir_style,oblique30_style")
    demo.add_argument("--style-refs", type=int, default=1)
    demo.add_argument("--seeds", type=int, default=1)
    demo.add_argument("--match-mode", default="edges", choices=["edges", "gray"])
    demo.add_argument("--min-inliers", type=int, default=15)
    demo.add_argument("--min-ratio", type=float, default=0.15)
    demo.add_argument("--min-coverage", type=float, default=0.03)
    demo.add_argument("--max-pairs", type=int, default=None)
    demo.add_argument("--clean", action="store_true")
    add_backend_args(demo)
    add_prompt_args(demo)
    demo.set_defaults(func=cmd_demo)

    sizing = sub.add_parser("sizing", help="estimate route, tiles, frames and generation cost")
    sizing.add_argument("--distance-km", type=float, default=5000.0)
    sizing.add_argument("--corridor-km", type=float, default=2.0)
    sizing.add_argument("--altitude-m", type=float, default=200.0)
    sizing.add_argument("--fov", type=float, default=73.0)
    sizing.add_argument("--speed-kmh", type=float, default=180.0)
    sizing.add_argument("--camera-fps", type=float, default=10.0)
    sizing.add_argument("--keyframe-hz", type=float, default=1.0)
    sizing.add_argument("--match-every-m", type=float, default=500.0)
    sizing.add_argument("--satellite-gsd-m", type=float, default=0.5)
    sizing.add_argument("--drone-pixels-wide", type=int, default=4000)
    sizing.add_argument("--tile-px", type=int, default=1024)
    sizing.add_argument("--gen-seconds", type=float, default=40.0)
    sizing.add_argument("--api-seconds", type=float, default=1.0)
    sizing.add_argument("--api-cost", type=float, default=0.009)
    sizing.add_argument("--gpus", type=int, default=1)
    sizing.add_argument("--match-seconds", type=float, default=0.3)
    sizing.add_argument("--frame-mb", type=float, default=1.0)
    sizing.add_argument("--generated-tile-mb", type=float, default=1.5)
    sizing.add_argument("--compression-ratio", type=float, default=4.0)
    sizing.add_argument("--quiet", action="store_true")
    sizing.set_defaults(func=cmd_sizing)

    selfcheck = sub.add_parser("selfcheck", help="verify matching/metrics on a synthetic pair")
    selfcheck.add_argument("--out", default="outputs/sat2aerial/selfcheck")
    selfcheck.add_argument("--match-mode", default="edges", choices=["edges", "gray"])
    selfcheck.add_argument("--seed", type=int, default=0)
    selfcheck.set_defaults(func=cmd_selfcheck)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
