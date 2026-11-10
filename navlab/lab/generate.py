from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from ..generation import generate_reference
from .tiles import TileStore


def generate_references(
    store: TileStore,
    out_dir: str | Path,
    *,
    backend: str = "dummy",
    view: str = "nadir",
    size: str = "768x768",
    seeds: int = 1,
    seed: int = 0,
    limit: int | None = None,
    style_refs: tuple[Path, ...] = (),
    name: str | None = None,
    altitude_m: float = 200.0,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    width_token, height_token = size.lower().split("x", 1)
    width = int(width_token)
    height = int(height_token)
    if width != height:
        raise ValueError("generated reference size must be square")
    variant_dir = out_dir / "generated" / view
    variant_dir.mkdir(parents=True, exist_ok=True)
    tiles = store.all_tiles()
    if limit is not None:
        tiles = tiles[: max(1, int(limit))]

    references: list[dict[str, Any]] = []
    for index, tile in enumerate(tiles):
        source_path = store.path_for(tile)
        for seed_offset in range(max(1, int(seeds))):
            tile_seed = int(seed) + index * 100 + seed_offset
            reference_id = f"{tile['id']}_{view}_s{tile_seed}"
            destination = variant_dir / f"{reference_id}.png"
            meta = generate_reference(
                satellite=source_path,
                out_path=destination,
                backend=backend,
                view=view,
                seed=tile_seed,
                size=size,
                altitude_m=altitude_m,
                style_refs=style_refs,
            )
            image = cv2.imread(str(destination), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(destination), image)
            bounds = dict(tile["bounds"])
            ground = float(store.tile_ground_m)
            gsd = ground / float(width)
            try:
                image_path = str(destination.relative_to(store.root))
            except ValueError:
                image_path = str(destination)
            references.append(
                {
                    "id": reference_id,
                    "image": image_path,
                    "east": float(tile["east"]),
                    "north": float(tile["north"]),
                    "bounds": bounds,
                    "lonlat": dict(tile.get("lonlat") or {}),
                    "gsd": gsd,
                    "tile_px": width,
                    "col": tile.get("col"),
                    "row": tile.get("row"),
                    "source": str(source_path),
                    "kind": "generated",
                    "variant": view,
                    "seed": tile_seed,
                    "source_tile": tile["id"],
                    "meta": meta,
                }
            )

    manifest = {
        "name": name or f"{store.root.name}_{view}",
        "kind": "generated",
        "variant": view,
        "backend": backend,
        "size": [width, height],
        "seeds": seeds,
        "base_seed": seed,
        "altitude_m": altitude_m,
        "style_refs": [str(path) for path in style_refs],
        "store": str(store.root),
        "tiles": references,
    }
    manifest_path = out_dir / f"references_{view}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["path"] = str(manifest_path)
    return manifest
