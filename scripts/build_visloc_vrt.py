#!/usr/bin/env python3
"""Build a read-only GDAL VRT from an aligned set of VisLoc GeoTIFFs."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import xml.etree.ElementTree as ET

import rasterio


def build_vrt(paths: list[Path], out: Path) -> None:
    if not paths:
        raise ValueError("no source rasters")
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in paths]
        first = sources[0]
        dx, dy = first.res
        if (first.crs is None or first.transform.a <= 0 or first.transform.e >= 0
                or first.transform.b != 0 or first.transform.d != 0 or dx <= 0 or dy <= 0):
            raise ValueError("expected a north-up georeferenced raster")
        for src in sources:
            if (src.crs != first.crs or src.res != first.res or src.count != first.count
                    or src.dtypes != first.dtypes or src.transform.b != 0 or src.transform.d != 0
                    or src.transform.a <= 0 or src.transform.e >= 0):
                raise ValueError("source grids, CRS, band counts and types must agree")
        if any(dtype != "uint8" for dtype in first.dtypes):
            raise ValueError("this VisLoc helper supports byte rasters only")
        left, top = min(s.bounds.left for s in sources), max(s.bounds.top for s in sources)
        right, bottom = max(s.bounds.right for s in sources), min(s.bounds.bottom for s in sources)
        width, height = round((right-left)/dx), round((top-bottom)/dy)
        root = ET.Element("VRTDataset", rasterXSize=str(width), rasterYSize=str(height))
        ET.SubElement(root, "SRS").text = first.crs.to_wkt()
        ET.SubElement(root, "GeoTransform").text = f"{left}, {dx}, 0, {top}, 0, {-dy}"
        for band in range(1, first.count + 1):
            element = ET.SubElement(root, "VRTRasterBand", dataType="Byte", band=str(band))
            if first.count == 3:
                ET.SubElement(element, "ColorInterp").text = ("Red", "Green", "Blue")[band-1]
            for path, src in zip(paths, sources):
                col, row = (src.bounds.left-left)/dx, (top-src.bounds.top)/dy
                if abs(col-round(col)) > 1e-5 or abs(row-round(row)) > 1e-5:
                    raise ValueError(f"raster is not aligned to the common grid: {path}")
                simple = ET.SubElement(element, "SimpleSource")
                ET.SubElement(simple, "SourceFilename", relativeToVRT="0").text = str(path.resolve())
                ET.SubElement(simple, "SourceBand").text = str(band)
                ET.SubElement(simple, "SrcRect", xOff="0", yOff="0", xSize=str(src.width), ySize=str(src.height))
                ET.SubElement(simple, "DstRect", xOff=str(round(col)), yOff=str(round(row)),
                              xSize=str(src.width), ySize=str(src.height))
        out.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/uav_visloc/UAV_VisLoc_dataset"))
    parser.add_argument("--map", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted((args.root / args.map).glob(f"satellite{args.map}_*.tif"))
    build_vrt(paths, args.out)
    print(f"Built {args.out} from {len(paths)} source rasters")


if __name__ == "__main__":
    main()
