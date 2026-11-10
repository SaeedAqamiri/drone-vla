#!/usr/bin/env python3
"""Verify same-query R/xDOP pairs and summarize their pose-test results."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    selection = json.loads(args.selection.read_text())
    rows = []
    for site, group in selection["sites"].items():
        for base in group["files"]:
            changed = base.replace("_R", "_xDOP")
            row = {"site": site, "same_domain": base, "cross_domain": changed}
            if (args.data/base).exists() and (args.data/changed).exists():
                with np.load(args.data/base, allow_pickle=False) as a, np.load(args.data/changed, allow_pickle=False) as b:
                    row["same_query"] = bool(np.array_equal(a["image_query"], b["image_query"]))
                    row["same_intrinsics"] = bool(np.array_equal(a["intrinsics"], b["intrinsics"]))
                    row["same_extrinsics"] = bool(np.array_equal(a["extrinsics"], b["extrinsics"]))
                    row["same_dsm"] = bool(np.array_equal(a["dsm"], b["dsm"], equal_nan=True))
                    row["same_dop"] = bool(np.array_equal(a["image_dop"], b["image_dop"]))
            for method in ("sift", "gim_dkm"):
                for variant, name in (("same", base), ("cross", changed)):
                    f = args.results / f"outplace_{Path(name).stem}_{method}_pnp.json"
                    if f.exists():
                        d = json.loads(f.read_text())
                        row[f"{method}_{variant}"] = {"success": d["success"],
                                                       "horizontal_error_m": d.get("horizontal_error_m")}
            rows.append(row)
    summary = {}
    for method in ("sift", "gim_dkm"):
        for variant in ("same", "cross"):
            key = f"{method}_{variant}"
            available = [r[key] for r in rows if key in r]
            summary[key] = {"processed": len(available), "accepted": sum(d["success"] for d in available),
                            "within_5m": sum(d["success"] and d.get("horizontal_error_m", float("inf")) <= 5 for d in available)}
    report = {"note": "Paired samples are correlated; compare same query, supplied map, DSM and intrinsics. "
                      "No flight trajectory or autonomous map retrieval is evaluated.", "summary": summary, "pairs": rows}
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
