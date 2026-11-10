#!/usr/bin/env python3
"""Choose evenly spaced OrthoLoC R samples per out-of-place site from its index."""
import argparse
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-site", type=int, default=5)
    a = p.parse_args()
    if a.per_site < 2:
        p.error("--per-site must be at least two")
    names = re.findall(r'href="(L\d+_R\d+\.npz)"', a.index.read_text())
    groups = {}
    for name in names:
        groups.setdefault(name.split("_")[0], []).append(name)
    selected = {}
    for site, group in sorted(groups.items()):
        group = sorted(set(group))
        if len(group) < a.per_site:
            raise ValueError(f"{site} has fewer than {a.per_site} files")
        selected[site] = {"population": len(group), "files": [
            group[round(i*(len(group)-1)/(a.per_site-1))] for i in range(a.per_site)]}
    if not selected:
        raise ValueError("no valid sample filenames in index")
    report = {"note": "Deterministic evenly spaced R files per site; no sample contents examined in selection.",
              "source": "https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/OrthoLoC/full/test_outPlace/",
              "sites": selected}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
