from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tiles import TileStore


def combine_references(
    store: TileStore,
    generated_manifest: str | Path,
    out_path: str | Path,
    *,
    name: str = "ensemble",
) -> dict[str, Any]:
    payload = json.loads(Path(generated_manifest).read_text(encoding="utf-8"))
    satellite_tiles: list[dict[str, Any]] = []
    for tile in store.all_tiles():
        item = dict(tile)
        item["kind"] = "satellite"
        satellite_tiles.append(item)
    generated_tiles = [dict(tile) for tile in payload.get("tiles", [])]
    for tile in generated_tiles:
        tile["kind"] = tile.get("kind") or "generated"
    combined = {
        "name": name,
        "kind": "ensemble",
        "store": str(store.root),
        "generated_manifest": str(generated_manifest),
        "satellite_references": len(satellite_tiles),
        "generated_references": len(generated_tiles),
        "tiles": satellite_tiles + generated_tiles,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return combined
