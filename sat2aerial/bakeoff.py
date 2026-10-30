from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .backends import Backend
from .dataset import pick_style_refs
from .matching import draw_matches, match_images
from .metrics import evaluate_pair
from .prompts import build_prompt
from .types import Pair

FIELDS = [
    "site",
    "variant",
    "view",
    "seed",
    "aerial",
    "match_mode",
    "good_matches",
    "inliers",
    "inlier_ratio",
    "median_reproj",
    "scale",
    "rotation_deg",
    "coverage",
    "success",
    "cons_inliers",
    "cons_ratio",
    "cons_coverage",
    "aligned_edge_iou",
    "ssim",
    "lab_mean_dist",
    "hist_corr",
    "seconds",
]

NUMERIC_FIELDS = [
    "good_matches",
    "inliers",
    "inlier_ratio",
    "median_reproj",
    "scale",
    "rotation_deg",
    "coverage",
    "success",
    "cons_inliers",
    "cons_ratio",
    "cons_coverage",
    "aligned_edge_iou",
    "ssim",
    "lab_mean_dist",
    "hist_corr",
    "seconds",
]


@dataclass
class BakeoffConfig:
    out_dir: Path
    variants: list[tuple[str, str, int]]
    size: tuple[int, int] = (768, 768)
    seed: int = 0
    seeds: int = 1
    match_mode: str = "edges"
    style_refs: int = 0
    min_inliers: int = 15
    min_ratio: float = 0.15
    min_coverage: float = 0.03
    tolerance_px: int = 3
    max_pairs: int | None = None
    altitude_m: int = 120
    season: str = "summer"
    lighting: str = "clear midday sunlight"
    weather: str = "clear"
    extra_prompt: str = ""
    max_dim: int = 1024


@dataclass
class BakeoffResult:
    out_dir: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    aggregate: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _label(image: np.ndarray, text: str, height: int = 420) -> np.ndarray:
    scale = height / float(image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)
    bar = np.zeros((30, resized.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, resized])


def _montage(paths: list[Path], labels: list[str], out_path: Path) -> bool:
    panels = []
    for path, label in zip(paths, labels):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return False
        panels.append(_label(image, label))
    if not panels:
        return False
    canvas = np.hstack(panels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), canvas))


def _mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    aggregate: dict[str, dict[str, Any]] = {}
    for variant, items in grouped.items():
        entry: dict[str, Any] = {"n": len(items)}
        for name in NUMERIC_FIELDS:
            entry[f"mean_{name}"] = _mean([item.get(name) for item in items])
        entry["success_rate"] = _mean([1.0 if item.get("success") else 0.0 for item in items])
        entry["mean_seconds"] = _mean([item.get("seconds") for item in items])
        successes = [1.0 if item.get("success") else 0.0 for item in items]
        inliers = [float(item["inliers"]) for item in items if item.get("inliers") is not None]
        entry["min_success"] = min(successes) if successes else None
        entry["max_success"] = max(successes) if successes else None
        entry["min_inliers"] = min(inliers) if inliers else None
        entry["max_inliers"] = max(inliers) if inliers else None
        entry["std_inliers"] = float(np.std(inliers)) if inliers else None
        entry["n_seeds"] = len({item.get("seed") for item in items if item.get("seed") is not None}) or 1
        aggregate[variant] = entry
    return aggregate


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_report(
    out_dir: Path,
    config: BakeoffConfig,
    backend: Backend,
    aggregate: dict[str, dict[str, Any]],
    n_pairs: int,
) -> None:
    lines = [
        "# Satellite to Aerial Bake-off",
        "",
        f"- backend: `{backend.name}` ({backend.model})",
        f"- pairs: {n_pairs}",
        f"- output size: {config.size[0]}x{config.size[1]}",
        f"- match mode: `{config.match_mode}`",
        f"- seeds per pair/variant: {config.seeds}",
        f"- success rule: inliers >= {config.min_inliers}, inlier_ratio >= {config.min_ratio}, "
        f"coverage >= {config.min_coverage}",
        "",
        "## Results",
        "",
        "| variant | n | success | inliers | ratio | coverage | consistency inliers | edge IoU | SSIM | Lab dist | hist corr | s/img |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = ["baseline", *[variant for variant in aggregate if variant != "baseline"]]
    for variant in order:
        entry = aggregate.get(variant)
        if not entry:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    str(entry["n"]),
                    _fmt(entry.get("success_rate"), 3),
                    _fmt(entry.get("mean_inliers"), 1),
                    _fmt(entry.get("mean_inlier_ratio"), 3),
                    _fmt(entry.get("mean_coverage"), 3),
                    _fmt(entry.get("mean_cons_inliers"), 1),
                    _fmt(entry.get("mean_aligned_edge_iou"), 3),
                    _fmt(entry.get("mean_ssim"), 3),
                    _fmt(entry.get("mean_lab_mean_dist"), 1),
                    _fmt(entry.get("mean_hist_corr"), 3),
                    _fmt(entry.get("mean_seconds"), 1),
                ]
            )
            + " |"
        )
    if config.seeds > 1:
        lines += [
            "",
            "## Seed variance (geometry hallucination check)",
            "",
            "| variant | seeds | min inliers | max inliers | std | worst success | best success |",
            "|---|---|---|---|---|---|---|",
        ]
        for variant, entry in aggregate.items():
            if variant == "baseline":
                continue
            lines.append(
                f"| {variant} | {entry.get('n_seeds')} | {_fmt(entry.get('min_inliers'), 1)} | "
                f"{_fmt(entry.get('max_inliers'), 1)} | {_fmt(entry.get('std_inliers'), 1)} | "
                f"{_fmt(entry.get('min_success'), 3)} | {_fmt(entry.get('max_success'), 3)} |"
            )
    baseline = aggregate.get("baseline")
    if baseline:
        lines += ["", "## Delta vs raw satellite baseline", "", "| variant | delta success | delta inliers |", "|---|---|---|"]
        for variant, entry in aggregate.items():
            if variant == "baseline":
                continue
            delta_success = (entry.get("success_rate") or 0.0) - (baseline.get("success_rate") or 0.0)
            delta_inliers = (entry.get("mean_inliers") or 0.0) - (baseline.get("mean_inliers") or 0.0)
            lines.append(f"| {variant} | {delta_success:+.3f} | {delta_inliers:+.1f} |")
    if backend.name == "dummy":
        lines += [
            "",
            "> Warning: the dummy backend is a procedural warp, not a generative model. "
            "These numbers only validate the harness and must not be used as evidence about FLUX 3.",
        ]
    lines += [
        "",
        "## How to read this",
        "",
        "- `success` / `inliers`: can the generated reference be matched to a real drone frame?",
        "- `consistency inliers`: does the generated image still match the source satellite tile "
        "(structure preservation)?",
        "- `edge IoU`: edge agreement after aligning the generated image back to the satellite.",
        "- A model that scores high on color but low on consistency is hallucinating plausible texture.",
        "",
        "Montages and match overlays are under `sites/`.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_bakeoff(
    pairs: list[Pair],
    backend: Backend,
    config: BakeoffConfig,
) -> BakeoffResult:
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sites_dir = out_dir / "sites"
    if config.max_pairs:
        pairs = pairs[: config.max_pairs]

    rows: list[dict[str, Any]] = []
    baseline_cache: dict[tuple[str, str], dict[str, Any]] = {}
    started_all = time.time()

    try:
        from tqdm import tqdm

        iterator = tqdm(pairs, desc="pairs", unit="pair")
    except Exception:
        iterator = pairs

    for pair_index, pair in enumerate(iterator):
        site_dir = sites_dir / _safe(pair.site_id)
        site_dir.mkdir(parents=True, exist_ok=True)

        for aerial in pair.aerials:
            key = (pair.site_id, str(aerial))
            if key not in baseline_cache:
                baseline, _ = match_images(
                    pair.satellite,
                    aerial,
                    mode=config.match_mode,
                    max_dim=config.max_dim,
                    min_inliers=config.min_inliers,
                    min_ratio=config.min_ratio,
                    min_coverage=config.min_coverage,
                )
                baseline_cache[key] = {
                    "site": pair.site_id,
                    "variant": "baseline",
                    "view": "none",
                    "seed": None,
                    "aerial": str(aerial),
                    "match_mode": config.match_mode,
                    **baseline.as_dict(),
                    "cons_inliers": None,
                    "cons_ratio": None,
                    "cons_coverage": None,
                    "aligned_edge_iou": None,
                    "ssim": None,
                    "lab_mean_dist": None,
                    "hist_corr": None,
                    "seconds": 0.0,
                }
            rows.append(baseline_cache[key])

        for variant_index, (variant_name, view, variant_style) in enumerate(config.variants):
            style_count = (
                config.style_refs if variant_style > 0 and config.style_refs > 0 else variant_style
            )
            style_refs = pick_style_refs(pairs, pair.site_id, style_count)
            prompt = build_prompt(
                view=view,
                style_refs=len(style_refs),
                altitude_m=config.altitude_m,
                season=config.season,
                lighting=config.lighting,
                weather=config.weather,
                extra=config.extra_prompt,
            )
            for seed_offset in range(config.seeds):
                seed = config.seed + pair_index * 1000 + variant_index * 100 + seed_offset
                suffix = "" if config.seeds == 1 else f"_s{seed_offset}"
                gen_path = site_dir / f"{_safe(variant_name)}{suffix}.png"
                seconds = 0.0
                if not gen_path.exists():
                    meta = backend.generate(
                        satellite=pair.satellite,
                        prompt=prompt,
                        out_path=gen_path,
                        size=config.size,
                        seed=seed,
                        style_refs=style_refs,
                    )
                    seconds = float(meta.get("seconds", 0.0) or 0.0)
                if not gen_path.exists():
                    raise RuntimeError(
                        f"Backend {backend.name!r} did not produce {gen_path}. "
                        "If this is a dry run, drop --dry-run for the bake-off."
                    )

                consistency, consistency_h = match_images(
                    gen_path,
                    pair.satellite,
                    mode="edges",
                    max_dim=config.max_dim,
                    min_inliers=config.min_inliers,
                    min_ratio=config.min_ratio,
                    min_coverage=config.min_coverage,
                )
                fidelity = evaluate_pair(gen_path, pair.satellite, consistency_h, config.tolerance_px)
                first_aerial = pair.aerials[0]
                draw_matches(
                    gen_path,
                    first_aerial,
                    site_dir / f"{_safe(variant_name)}{suffix}_matches.png",
                    mode=config.match_mode,
                    max_dim=config.max_dim,
                )
                _montage(
                    [pair.satellite, gen_path, first_aerial],
                    ["satellite", f"generated ({variant_name}{suffix})", "real aerial"],
                    site_dir / f"{_safe(variant_name)}{suffix}_montage.png",
                )

                for aerial in pair.aerials:
                    match, _ = match_images(
                        gen_path,
                        aerial,
                        mode=config.match_mode,
                        max_dim=config.max_dim,
                        min_inliers=config.min_inliers,
                        min_ratio=config.min_ratio,
                    min_coverage=config.min_coverage,
                    )
                    rows.append(
                        {
                            "site": pair.site_id,
                            "variant": variant_name,
                            "view": view,
                            "seed": seed,
                            "aerial": str(aerial),
                            "match_mode": config.match_mode,
                            **match.as_dict(),
                            "cons_inliers": consistency.inliers,
                            "cons_ratio": round(consistency.inlier_ratio, 4),
                            "cons_coverage": round(consistency.coverage, 4),
                            "aligned_edge_iou": None
                            if fidelity["aligned_edge_iou"] is None
                            else round(fidelity["aligned_edge_iou"], 4),
                            "ssim": None if fidelity["ssim"] is None else round(fidelity["ssim"], 4),
                            "lab_mean_dist": None
                            if fidelity["lab_mean_dist"] is None
                            else round(fidelity["lab_mean_dist"], 3),
                            "hist_corr": None
                            if fidelity["hist_corr"] is None
                            else round(fidelity["hist_corr"], 4),
                            "seconds": round(seconds, 3),
                        }
                    )

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    aggregate = _aggregate(rows)
    summary = {
        "backend": {"name": backend.name, "model": backend.model},
        "config": {
            "size": list(config.size),
            "seed": config.seed,
            "seeds": config.seeds,
            "match_mode": config.match_mode,
            "success_rule": {
                "min_inliers": config.min_inliers,
                "min_ratio": config.min_ratio,
                "min_coverage": config.min_coverage,
            },
            "variants": [list(item) for item in config.variants],
            "pairs": len(pairs),
            "aerial_frames": sum(len(pair.aerials) for pair in pairs),
        },
        "aggregate": aggregate,
        "wall_seconds": round(time.time() - started_all, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(out_dir, config, backend, aggregate, len(pairs))
    return BakeoffResult(out_dir=out_dir, rows=rows, aggregate=aggregate, summary=summary)
