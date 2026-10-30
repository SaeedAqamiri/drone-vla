from __future__ import annotations

import csv
from pathlib import Path

from .types import Pair

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
SAT_KEYS = ("satellite", "sat", "map", "reference", "ref")
AERIAL_KEYS = ("aerial", "drone", "uav", "query", "frame", "camera")


def _images_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _matches_stem(path: Path, keys: tuple[str, ...]) -> bool:
    stem = path.stem.lower()
    return any(stem == k or stem.startswith(k + "_") or stem.startswith(k + "-") for k in keys)


def _pick_satellite(files: list[Path]) -> Path | None:
    for path in files:
        if _matches_stem(path, SAT_KEYS):
            return path
    return None


def _pick_aerials(files: list[Path], satellite: Path) -> list[Path]:
    chosen = [p for p in files if p != satellite and _matches_stem(p, AERIAL_KEYS)]
    if chosen:
        return chosen
    return [p for p in files if p != satellite]


def _load_manifest(path: Path) -> list[Pair]:
    base = path.parent
    pairs: list[Pair] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"satellite"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain a 'satellite' column")
        for index, row in enumerate(reader):
            sat_raw = (row.get("satellite") or "").strip()
            if not sat_raw:
                continue
            satellite = Path(sat_raw)
            if not satellite.is_absolute():
                satellite = base / satellite
            aerial_raw = (row.get("aerial") or row.get("aerials") or "").strip()
            aerials = [
                (Path(item.strip()) if Path(item.strip()).is_absolute() else base / item.strip())
                for item in aerial_raw.replace("|", ";").split(";")
                if item.strip()
            ]
            site_id = (row.get("site") or row.get("site_id") or f"pair_{index:04d}").strip()
            meta = {k: v for k, v in row.items() if k not in {"site", "site_id", "satellite", "aerial", "aerials"}}
            pairs.append(Pair(site_id=site_id, satellite=satellite, aerials=aerials, meta=meta))
    return pairs


def _load_flat(root: Path) -> list[Pair]:
    sat_dir = root / "satellite"
    aerial_dir = root / "aerial"
    if not sat_dir.is_dir() or not aerial_dir.is_dir():
        return []
    aerials_by_stem: dict[str, list[Path]] = {}
    for path in _images_in(aerial_dir):
        aerials_by_stem.setdefault(path.stem.lower(), []).append(path)
    pairs: list[Pair] = []
    for sat in _images_in(sat_dir):
        stem = sat.stem.lower()
        matched = aerials_by_stem.get(stem, [])
        if not matched:
            matched = [
                path
                for key, values in aerials_by_stem.items()
                if key.startswith(stem)
                for path in values
            ]
        if matched:
            pairs.append(Pair(site_id=sat.stem, satellite=sat, aerials=matched))
    return pairs


def _load_subdirs(root: Path) -> list[Pair]:
    pairs: list[Pair] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        files = _images_in(child)
        if not files:
            continue
        satellite = _pick_satellite(files)
        if satellite is None:
            continue
        pairs.append(
            Pair(
                site_id=child.name,
                satellite=satellite,
                aerials=_pick_aerials(files, satellite),
            )
        )
    return pairs


def load_pairs(root: str | Path) -> list[Pair]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        return _load_manifest(root)
    manifest = root / "manifest.csv"
    if manifest.is_file():
        pairs = _load_manifest(manifest)
    else:
        pairs = _load_flat(root) or _load_subdirs(root)
    pairs = [p for p in pairs if p.satellite.exists() and p.aerials]
    if not pairs:
        raise ValueError(
            f"No pairs found under {root}. Expected <site>/satellite.png + aerial*.png, "
            "a manifest.csv, or satellite/ + aerial/ folders."
        )
    return pairs


def style_reference_pool(pairs: list[Pair]) -> dict[str, list[Path]]:
    return {pair.site_id: list(pair.aerials) for pair in pairs}


def pick_style_refs(
    pairs: list[Pair],
    site_id: str,
    count: int,
) -> list[Path]:
    if count <= 0:
        return []
    ordered = [p for p in pairs if p.site_id != site_id]
    chosen: list[Path] = []
    index = 0
    while len(chosen) < count and ordered:
        pair = ordered[index % len(ordered)]
        if pair.aerials:
            chosen.append(pair.aerials[(index // len(ordered)) % len(pair.aerials)])
        index += 1
        if index > len(ordered) * 8:
            break
    return chosen[:count]
