#!/usr/bin/env python3
"""Import and score every flight in a UAV-VisLoc-style dataset in one pass.

Wraps ``import_uav_visloc`` and ``sat2aerial.localize`` so a whole download can
be benchmarked without hand-running two commands per flight::

    python scripts/run_uav_visloc.py \
        --src data/uav_visloc/raw_v1 --out data/uav_visloc/processed_v1

The v1 product ships a north-up GeoTIFF, while the drone frames are mounted
~90 degrees off, so v1 flights get a rotation search; v2 flights are already
north-up and use a plain correlation match.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for entry in (REPO, REPO / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import import_uav_visloc  # noqa: E402
from sat2aerial.localize import LocalizeConfig, evaluate  # noqa: E402


def convert(site: Path, out: Path, args: argparse.Namespace) -> dict:
    if import_uav_visloc.is_v2(site):
        return import_uav_visloc.convert_flight_v2(
            site=site,
            out=out,
            ref_size=args.ref_size,
            ref_meters=args.ref_meters,
            ref_aspect=args.ref_aspect,
            query_size=args.query_size,
            max_frames=args.max_frames,
            offset_m=(0.0, 0.0),
            jpeg_quality=95,
            frame_gsd_arg=args.frame_gsd,
            strip_shift_arg="auto",
            frame_stride=args.frame_stride,
            start_frame=args.start_frame,
        )
    return import_uav_visloc.convert_flight(
        site=site,
        out=out,
        ref_meters=args.ref_meters,
        ref_size=args.ref_size,
        ref_aspect=args.ref_aspect,
        query_size=args.query_size,
        max_frames=args.max_frames,
        offset_m=(0.0, 0.0),
        jpeg_quality=95,
        frame_gsd_arg=args.frame_gsd,
        gsd_mode=args.gsd_mode,
        frame_stride=args.frame_stride,
        start_frame=args.start_frame,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sites", default="", help="comma separated flight ids (default: all)")
    parser.add_argument("--ref-meters", type=float, default=0.0)
    parser.add_argument("--ref-aspect", type=float, default=0.0)
    parser.add_argument("--ref-size", type=int, default=0)
    parser.add_argument("--query-size", type=int, default=0)
    parser.add_argument("--frame-gsd", default="auto")
    parser.add_argument("--gsd-mode", default="native", choices=["native", "square"])
    parser.add_argument("--matcher", default="ncc", choices=["ncc", "features"])
    parser.add_argument("--match-mode", default="edges", choices=["gray", "edges"])
    parser.add_argument("--min-score", type=float, default=0.12)
    parser.add_argument("--rotate-search", default="auto", help="'auto', 'off', or degrees")
    parser.add_argument("--rotate-step", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1, help="keep every Nth frame")
    parser.add_argument("--start-frame", type=int, default=0, help="skip the first N frames")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)
    wanted = {token.strip() for token in args.sites.split(",") if token.strip()}
    sites = []
    for child in sorted(src.iterdir()):
        if not child.is_dir() or (wanted and child.name not in wanted):
            continue
        if import_uav_visloc.is_v2(child) or import_uav_visloc._find_site_files(child)[0] is not None:
            sites.append(child)
    if import_uav_visloc.is_v2(src) or (src / "drone").is_dir():
        sites = [src]
    if not sites:
        raise SystemExit(f"no flights under {src}")

    total = len(sites)
    results = []
    for index, site in enumerate(sites, start=1):
        destination = out / site.name if total > 1 else out
        print(f"[{index}/{total}] {site.name} -> {destination}", flush=True)
        if destination.exists() and any(destination.iterdir()) and not args.overwrite:
            print("  exists, use --overwrite; skipping", flush=True)
            continue
        try:
            meta = convert(site, destination, args)
        except Exception as error:  # noqa: BLE001 - one bad flight must not stop the sweep
            print(f"  import failed: {error}", flush=True)
            results.append({"site": site.name, "error": str(error)})
            continue

        if args.matcher == "ncc":
            rotate = args.rotate_search
            if rotate == "auto":
                rotate = 0.0 if meta["layout"] == "uav_visloc_v2" else 360.0
            elif rotate == "off":
                rotate = 0.0
            else:
                rotate = float(rotate)
            config = LocalizeConfig(
                run=destination,
                out=destination / "eval",
                matcher="ncc",
                min_score=args.min_score,
                rotate_search=rotate,
                rotate_step=args.rotate_step,
                max_frames=args.max_frames or None,
            )
        else:
            config = LocalizeConfig(
                run=destination,
                out=destination / "eval",
                matcher="features",
                mode=args.match_mode,
                max_frames=args.max_frames or None,
            )

        try:
            summary = evaluate(config)
        except Exception as error:  # noqa: BLE001
            print(f"  evaluation failed: {error}", flush=True)
            results.append({"site": site.name, "error": str(error)})
            continue

        row = {
            "site": site.name,
            "layout": meta["layout"],
            "frames": summary["frames"],
            "fix_rate": summary["fix_rate"],
            "median_m": summary["error_m"]["median"],
            "p90_m": summary["error_m"]["p90"],
            "bias_m": summary["bias_m"]["magnitude"] if "bias_m" in summary else None,
            "debiased_median_m": (
                summary["error_m_debiased"]["median"] if "error_m_debiased" in summary else None
            ),
            "under_2m": summary.get("under_2m"),
            "under_5m": summary.get("under_5m"),
            "gross_5m": summary["gated"]["gross_passed_5m"],
        }
        results.append(row)
        print(
            f"  fix {row['fix_rate']:.2f} | median {row['median_m']} m | "
            f"p90 {row['p90_m']} m | debiased {row['debiased_median_m']} m",
            flush=True,
        )

    out.mkdir(parents=True, exist_ok=True)
    previous = []
    if (out / "summary.json").is_file():
        try:
            previous = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = []
    merged = {row.get("site"): row for row in previous if isinstance(row, dict)}
    for row in results:
        merged[row.get("site")] = row
    results = [merged[key] for key in sorted(merged) if key]
    (out / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["site", "layout", "frames", "fix_rate", "median_m", "p90_m", "bias_m",
              "debiased_median_m", "under_2m", "under_5m", "gross_5m", "error"]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nwrote {out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
