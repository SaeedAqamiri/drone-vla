#!/usr/bin/env python3
"""Align and filter generated tiles against their original satellite tiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sat2aerial.geometry import register_generated_to_source
from navlab.lab.tiles import TileStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    store = TileStore(args.store)
    payload = json.loads(Path(args.references).read_text())
    source_by_id = {t["id"]: t for t in store.tiles}
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    selected, diagnostics = [], []
    for item in payload["tiles"]:
        src = source_by_id.get(item.get("source_tile"))
        if src is None:
            diagnostics.append({"id": item["id"], "accepted": False, "reason": "source_missing"})
            continue
        source = cv2.imread(str(store.path_for(src)))
        image = Path(item["image"])
        if not image.is_absolute():
            image = store.root / image
        generated = cv2.imread(str(image))
        aligned, diag = register_generated_to_source(source, generated)
        diag["id"] = item["id"]
        diagnostics.append(diag)
        if aligned is None:
            continue
        path = (output / (item["id"] + ".png")).resolve()
        cv2.imwrite(str(path), aligned)
        selected.append({**item, "image": str(path), "geometry_check": diag})
    result = {**payload, "name": payload.get("name", "generated") + "_geometry_checked",
              "tiles": selected, "geometry_filter": {"total": len(diagnostics),
              "accepted": len(selected), "note": "Not a proof of local geometric fidelity."}}
    (output / "references_validated.json").write_text(json.dumps(result, indent=2))
    (output / "geometry_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(result["geometry_filter"], indent=2))


if __name__ == "__main__":
    main()
