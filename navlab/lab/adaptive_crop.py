"""Experimental prior-centred map crops; footprint and uncertainty are inputs.

A crop is selected without evaluation labels. The footprint and prior sigma
are heuristic unless a caller supplies calibrated estimates; no coverage or
flight-safety guarantee follows from them.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT


class AdaptiveCrop:
    def __init__(self, store, *, footprint_width_m: float, uncertainty_scale: float = 2.0,
                 max_side_m: float = 1500.0, pixels: int = 1024, min_valid_fraction: float = .95,
                 sigma_growth_mps: float = 3.0, horizontal_fov_deg: float | None = None):
        if (not math.isfinite(footprint_width_m) or footprint_width_m <= 0
                or not math.isfinite(uncertainty_scale) or uncertainty_scale < 0
                or not math.isfinite(max_side_m) or max_side_m <= 0
                or pixels < 64 or not 0 < min_valid_fraction <= 1
                or not math.isfinite(sigma_growth_mps) or sigma_growth_mps < 0
                or (horizontal_fov_deg is not None and (not math.isfinite(horizontal_fov_deg)
                                                         or not 0 < horizontal_fov_deg < 170))):
            raise ValueError('invalid adaptive crop configuration')
        source = Path(store.manifest['source'])
        self.src = rasterio.open(source)
        if self.src.count < 3 or not self.src.crs or any(dtype != 'uint8' for dtype in self.src.dtypes[:3]):
            self.src.close()
            raise ValueError('adaptive crop requires a georeferenced 8-bit RGB source')
        self.crs = store.crs
        self.footprint_width_m = float(footprint_width_m)
        self.uncertainty_scale = float(uncertainty_scale)
        self.max_side_m = float(max_side_m)
        self.pixels = int(pixels)
        self.min_valid_fraction = float(min_valid_fraction)
        self.sigma_growth_mps = float(sigma_growth_mps)
        self.horizontal_fov_deg = horizontal_fov_deg
        self.last_diagnostic = {}

    def close(self):
        self.src.close()

    def crop(self, east: float, north: float, sigma: float, image_shape: tuple[int, int], *,
             altitude_agl_m: float | None = None):
        h, w = image_shape
        if h <= 0 or w <= 0 or not all(math.isfinite(x) for x in (east, north, sigma)) or sigma < 0:
            raise ValueError('invalid crop position, uncertainty, or frame shape')
        # With unknown heading, a square around the full image diagonal contains
        # every rotation of a planar, near-nadir image of the supplied width.
        footprint_width = self.footprint_width_m
        if self.horizontal_fov_deg is not None and altitude_agl_m is not None:
            if not math.isfinite(altitude_agl_m) or altitude_agl_m <= 0:
                raise ValueError("altitude above ground must be positive and finite")
            footprint_width = 2 * altitude_agl_m * math.tan(math.radians(self.horizontal_fov_deg) / 2)
        diag = footprint_width * math.hypot(w, h) / w
        side = diag + 2 * self.uncertainty_scale * sigma
        self.last_diagnostic = {"side_m": side, "sigma_m": sigma, "footprint_width_m": footprint_width,
                                "uses_altitude_agl": self.horizontal_fov_deg is not None and altitude_agl_m is not None,
                                "rejection": None}
        if side > self.max_side_m:
            self.last_diagnostic["rejection"] = "max_side_exceeded"
            return None
        half = side / 2
        bounds = {'min_east':east-half, 'min_north':north-half,
                  'max_east':east+half, 'max_north':north+half}
        transform = from_bounds(bounds['min_east'], bounds['min_north'],
                                bounds['max_east'], bounds['max_north'], self.pixels, self.pixels)
        # Alpha identifies places outside the available map. Never silently
        # treat an incomplete map window as a valid complete candidate.
        with WarpedVRT(self.src, crs=self.crs, transform=transform,
                       width=self.pixels, height=self.pixels,
                       add_alpha=True, resampling=Resampling.bilinear) as vrt:
            data = vrt.read([1, 2, 3])
            valid_mask = vrt.read(vrt.count)
            valid_fraction = float(np.count_nonzero(valid_mask) / (self.pixels ** 2))
        self.last_diagnostic["valid_fraction"] = valid_fraction
        if valid_fraction < self.min_valid_fraction:
            self.last_diagnostic["rejection"] = "insufficient_map_fraction"
            return None
        image = cv2.cvtColor(np.moveaxis(data, 0, -1), cv2.COLOR_RGB2BGR)
        tile = {'id':'adaptive_crop', 'bounds':bounds, 'gsd':side/self.pixels,
                'tile_px':self.pixels, 'kind':'satellite', '_image':image,
                'valid_fraction':valid_fraction, 'side_m':side, '_valid_mask':valid_mask}
        return tile
