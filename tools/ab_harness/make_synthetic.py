#!/usr/bin/env python
"""Write a small deterministic processed_inputs/ folder for the harness, so it runs
without private data. A synthetic 400 x 400 px city at 2 m in UTM 43N (EPSG:32643) is
tiled the way preprocess(tile_size=200, overlap=20) tiles a raster: keys are the
x_y pixel offsets 0_0, 0_200, 200_0 and 200_200, and each tile extends 20 px past its
200 px core to the right and bottom where the raster allows (220 x 220, 220 x 200,
200 x 220 and 200 x 200 px).

The city: a gently sloped DEM, a block grid of 6-45 m buildings (several straddling
the tile borders), an 80 m tower across the central tile corner and 5-14 m tree
clumps. The seed is fixed, so every run writes the same rasters.

--bush adds a 3 x 3 patch of phantom bushes to every tile: 3.0 m trees over a -0.75 m
DEM, where 0.25 * trees + DEM == 0 (UPSTREAM_CHANGES.md section 5, quirk 3). This sends
svf_calculator down the per-patch shadow() path (section 3.6). Trees older than
upstream v2.0.0 (e.g. 38b451d) crash on such tiles; use v2.0.0 or later as reference.

Usage: make_synthetic.py OUT [--bush]
"""
import argparse
import os

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

SIZE, TILE, OVERLAP = 400, 200, 20
GT = (276923.4, 2.0, 0.0, 2109921.7, 0.0, -2.0)  # UTM 43N, near Mumbai
SEED = 20261104


def city(bush):
    """Building DSM, DEM and trees of the whole city (float32)."""
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    dem = (5.0 + 0.004 * xx + 0.006 * yy).astype(np.float32)  # gentle slope, > 0
    bld = np.zeros((SIZE, SIZE), np.float32)
    for by in range(10, SIZE - 30, 48):
        for bx in range(10, SIZE - 30, 48):
            if rng.random() < 0.15:
                continue  # open plot
            dy, dx = (int(v) for v in rng.integers(14, 34, size=2))
            bld[by:by + dy, bx:bx + dx] = float(rng.uniform(6, 45))
    bld[185:215, 185:215] = 80.0  # tower across the central tile corner
    trees = np.zeros((SIZE, SIZE), np.float32)
    for _ in range(60):
        cy, cx = (int(v) for v in rng.integers(0, SIZE, size=2))
        r, h = int(rng.integers(3, 9)), float(rng.uniform(5, 14))
        m = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r) & (bld == 0)
        trees[m] = np.maximum(trees[m], h)
    if bush:
        for y0 in range(0, SIZE, TILE):
            for x0 in range(0, SIZE, TILE):
                y, x = open_spot(bld, trees, y0 + TILE // 2, x0 + TILE // 2)
                trees[y:y + 3, x:x + 3] = 3.0
                dem[y:y + 3, x:x + 3] = -0.75
    return dem + bld, dem, trees


def open_spot(bld, trees, cy, cx):
    """Top-left corner of the 3 x 3 window of open ground (no building, no tree)
    nearest to (cy, cx), searching outwards ring by ring."""
    for r in range(60):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dy), abs(dx)) != r:
                    continue
                win = (slice(cy + dy, cy + dy + 3), slice(cx + dx, cx + dx + 3))
                if not bld[win].any() and not trees[win].any():
                    return cy + dy, cx + dx
    raise RuntimeError(f"no open ground near ({cy}, {cx})")


def write(path, arr, x0, y0):
    """Write one tile whose top-left pixel is (x0, y0) of the city grid."""
    h, w = arr.shape
    ds = gdal.GetDriverByName("GTiff").Create(path, w, h, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((GT[0] + x0 * GT[1], GT[1], 0.0, GT[3] + y0 * GT[5], 0.0, GT[5]))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32643)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(arr)
    ds = None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="output folder; pass it to the other scripts as --data")
    ap.add_argument("--bush", action="store_true",
                    help="add a phantom-bush patch to every tile (per-patch shadow path)")
    args = ap.parse_args()

    dsm, dem, trees = city(args.bush)
    rasters = {"Building_DSM": dsm, "DEM": dem, "Trees": trees}
    keys = []
    for name in rasters:
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
    for x0 in range(0, SIZE, TILE):
        for y0 in range(0, SIZE, TILE):
            w, h = min(TILE + OVERLAP, SIZE - x0), min(TILE + OVERLAP, SIZE - y0)
            key = f"{x0}_{y0}"
            for name, arr in rasters.items():
                write(os.path.join(args.out, name, f"{name}_{key}.tif"),
                      arr[y0:y0 + h, x0:x0 + w], x0, y0)
            keys.append(key)
    print(f"wrote tiles {', '.join(keys)} to {args.out}: buildings "
          f"{int((dsm - dem > 0).sum())} px (max {float((dsm - dem).max()):.0f} m), trees "
          f"{int((trees > 0).sum())} px, phantom bushes {int((dem < 0).sum())} px")


if __name__ == "__main__":
    main()
