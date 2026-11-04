from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .bench import run_bench
from .generate import generate_references
from .references import combine_references
from .report import write_run
from .runner import LabRunner, RunnerConfig
from .sequences import (
    load_queries,
    route_lonlat,
    uav_visloc_queries,
    uav_visloc_satellite,
    video_queries,
    write_queries,
)
from .tiles import TileStore, build_tiles_along_route, build_tiles_in_bbox


def _parse_bbox(value: str) -> dict[str, float]:
    parts = [float(item) for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be min_lon,min_lat,max_lon,max_lat")
    return {
        "min_lon": min(parts[0], parts[2]),
        "min_lat": min(parts[1], parts[3]),
        "max_lon": max(parts[0], parts[2]),
        "max_lat": max(parts[1], parts[3]),
    }


def _read_route_csv(path: Path, lat_col: str, lon_col: str) -> list[tuple[float, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        route = []
        for row in reader:
            route.append((float(row[lon_col]), float(row[lat_col])))
    return route


def _cmd_tile_store(args: argparse.Namespace) -> int:
    if args.bbox:
        manifest = build_tiles_in_bbox(
            args.source,
            args.out,
            _parse_bbox(args.bbox),
            gsd=args.gsd,
            tile_px=args.tile_px,
            tile_ground_m=args.tile_ground_m,
            stride_m=args.stride_m,
            max_tiles=args.max_tiles,
            prefix=args.prefix,
        )
    else:
        if args.queries:
            queries, _ = load_queries(args.queries)
            route = route_lonlat(queries)
        elif args.route_csv:
            route = _read_route_csv(Path(args.route_csv), args.lat_col, args.lon_col)
        else:
            raise SystemExit("provide --route-csv, --queries, or --bbox")
        manifest = build_tiles_along_route(
            args.source,
            args.out,
            route,
            gsd=args.gsd,
            tile_px=args.tile_px,
            tile_ground_m=args.tile_ground_m,
            stride_m=args.stride_m,
            max_tiles=args.max_tiles,
            prefix=args.prefix,
            offset_east_m=args.offset_east,
            offset_north_m=args.offset_north,
        )
    print(f"tiles: {len(manifest['tiles'])}")
    print(f"manifest: {Path(args.out) / 'manifest.json'}")
    return 0


def _cmd_sequence(args: argparse.Namespace) -> int:
    if args.dataset != "uav_visloc":
        raise SystemExit("only --dataset uav_visloc is implemented")
    queries = uav_visloc_queries(
        args.root,
        args.map,
        near_nadir_deg=args.near_nadir,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    satellite = uav_visloc_satellite(args.root, args.map)
    meta = {
        "dataset": "uav_visloc",
        "map": args.map,
        "satellite": str(satellite),
        "near_nadir_deg": args.near_nadir,
        "stride": args.stride,
        "count": len(queries),
    }
    path = write_queries(args.out, queries, meta)
    print(f"queries: {len(queries)}")
    print(f"manifest: {path}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    store = TileStore(args.store)
    manifest = generate_references(
        store,
        args.out,
        backend=args.backend,
        view=args.view,
        size=args.size,
        seeds=args.seeds,
        seed=args.seed,
        limit=args.limit,
        altitude_m=args.altitude,
    )
    print(f"references: {len(manifest['tiles'])}")
    print(f"manifest: {manifest['path']}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    store = TileStore(args.store)
    queries, meta = load_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    config = RunnerConfig(
        matcher=args.matcher,
        mode=args.mode,
        max_dim=args.max_dim,
        max_candidates=args.max_candidates,
        bootstrap_candidates=args.bootstrap_candidates,
        base_radius=args.base_radius,
        max_radius=args.max_radius,
        attempt_every=args.attempt_every,
        max_jump_m=args.max_jump_m,
        scale_tolerance=args.scale_tolerance,
        camera_fov_deg=args.camera_fov,
        dropout_start=args.dropout_start,
        dropout_length=args.dropout_length,
        reference_kind=args.reference_kind,
        confirmation=args.confirmation,
        confirmation_radius_m=args.confirmation_radius,
        rng_seed=args.rng_seed,
    )
    runner = LabRunner(
        store,
        queries,
        config,
        references=args.references,
    )
    result = runner.run()
    reference_manifest = (
        json.loads(Path(args.references).read_text(encoding="utf-8"))
        if args.references
        else None
    )
    out_dir = write_run(
        args.out,
        result,
        store=store,
        reference_manifest=reference_manifest,
        extra={"sequence": meta},
    )
    print(f"run: {out_dir}")
    print(json.dumps(result.summary, indent=2))
    return 0


def _cmd_combine(args: argparse.Namespace) -> int:
    store = TileStore(args.store)
    payload = combine_references(
        store,
        args.references,
        args.out,
        name=args.name,
    )
    print(f"references: {len(payload['tiles'])}")
    print(f"manifest: {args.out}")
    return 0


def _cmd_video(args: argparse.Namespace) -> int:
    queries = video_queries(
        args.video,
        args.out,
        telemetry_csv=args.telemetry,
        sample_fps=args.fps,
        time_column=args.time_col,
        lat_column=args.lat_col,
        lon_column=args.lon_col,
        altitude_column=args.alt_col,
        yaw_column=args.yaw_col,
        time_offset=args.time_offset,
    )
    print(f"queries: {len(queries)}")
    print(f"manifest: {Path(args.out) / 'queries.json'}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    out_dir = run_bench(args.config, args.out)
    print(f"bench: {out_dir}")
    print(f"report: {out_dir / 'report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m navlab.lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tile_parser = subparsers.add_parser("tile-store", help="build metric reference tiles")
    tile_parser.add_argument("--source", required=True)
    tile_parser.add_argument("--out", required=True)
    tile_parser.add_argument("--route-csv")
    tile_parser.add_argument("--queries")
    tile_parser.add_argument("--lat-col", default="lat")
    tile_parser.add_argument("--lon-col", default="lon")
    tile_parser.add_argument("--bbox")
    tile_parser.add_argument("--gsd", type=float)
    tile_parser.add_argument("--tile-px", type=int, default=768)
    tile_parser.add_argument("--tile-ground-m", type=float)
    tile_parser.add_argument("--stride-m", type=float)
    tile_parser.add_argument("--max-tiles", type=int)
    tile_parser.add_argument("--prefix")
    tile_parser.add_argument("--offset-east", type=float, default=0.0)
    tile_parser.add_argument("--offset-north", type=float, default=0.0)
    tile_parser.set_defaults(func=_cmd_tile_store)

    sequence_parser = subparsers.add_parser("sequence", help="export a query sequence")
    sequence_parser.add_argument("--dataset", default="uav_visloc")
    sequence_parser.add_argument("--root", default="data/raw/uav_visloc/UAV_VisLoc_dataset")
    sequence_parser.add_argument("--map", required=True)
    sequence_parser.add_argument("--near-nadir", type=float, default=15.0)
    sequence_parser.add_argument("--stride", type=int, default=1)
    sequence_parser.add_argument("--max-frames", type=int)
    sequence_parser.add_argument("--out", required=True)
    sequence_parser.set_defaults(func=_cmd_sequence)

    video_parser = subparsers.add_parser("video", help="extract query frames from a video")
    video_parser.add_argument("--video", required=True)
    video_parser.add_argument("--out", required=True)
    video_parser.add_argument("--telemetry")
    video_parser.add_argument("--fps", type=float, default=2.0)
    video_parser.add_argument("--time-col", default="timestamp")
    video_parser.add_argument("--lat-col", default="lat")
    video_parser.add_argument("--lon-col", default="lon")
    video_parser.add_argument("--alt-col", default="alt")
    video_parser.add_argument("--yaw-col", default="yaw")
    video_parser.add_argument("--time-offset", type=float, default=0.0)
    video_parser.set_defaults(func=_cmd_video)

    generate_parser = subparsers.add_parser("generate", help="generate aerial references")
    generate_parser.add_argument("--store", required=True)
    generate_parser.add_argument("--out", required=True)
    generate_parser.add_argument("--backend", default="dummy")
    generate_parser.add_argument("--view", default="nadir")
    generate_parser.add_argument("--size", default="768x768")
    generate_parser.add_argument("--seeds", type=int, default=1)
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--limit", type=int)
    generate_parser.add_argument("--altitude", type=float, default=200.0)
    generate_parser.set_defaults(func=_cmd_generate)

    combine_parser = subparsers.add_parser(
        "combine", help="combine satellite and generated references"
    )
    combine_parser.add_argument("--store", required=True)
    combine_parser.add_argument("--references", required=True)
    combine_parser.add_argument("--out", required=True)
    combine_parser.add_argument("--name", default="ensemble")
    combine_parser.set_defaults(func=_cmd_combine)

    run_parser = subparsers.add_parser("run", help="run a localization benchmark")
    run_parser.add_argument("--store", required=True)
    run_parser.add_argument("--queries", required=True)
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--references")
    run_parser.add_argument("--matcher", default="sift")
    run_parser.add_argument("--mode", default="edges")
    run_parser.add_argument("--max-dim", type=int, default=1024)
    run_parser.add_argument("--max-candidates", type=int, default=8)
    run_parser.add_argument("--bootstrap-candidates", type=int, default=20)
    run_parser.add_argument("--base-radius", type=float, default=60.0)
    run_parser.add_argument("--max-radius", type=float, default=600.0)
    run_parser.add_argument("--attempt-every", type=int, default=1)
    run_parser.add_argument("--max-jump-m", type=float, default=90.0)
    run_parser.add_argument("--scale-tolerance", type=float, default=0.5)
    run_parser.add_argument("--camera-fov", type=float, default=73.0)
    run_parser.add_argument("--dropout-start", type=int)
    run_parser.add_argument("--dropout-length", type=int, default=0)
    run_parser.add_argument("--reference-kind")
    run_parser.add_argument("--confirmation", type=int, default=1)
    run_parser.add_argument("--confirmation-radius", type=float, default=200.0)
    run_parser.add_argument("--rng-seed", type=int, default=0)
    run_parser.add_argument("--limit", type=int)
    run_parser.set_defaults(func=_cmd_run)

    bench_parser = subparsers.add_parser("bench", help="run a matrix of lab configs")
    bench_parser.add_argument("--config", required=True)
    bench_parser.add_argument("--out", required=True)
    bench_parser.set_defaults(func=_cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
