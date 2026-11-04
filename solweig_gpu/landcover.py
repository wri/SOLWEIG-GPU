#SOLWEIG-GPU: GPU-accelerated SOLWEIG model for urban thermal comfort simulation
#Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
"""
Land cover inputs: a check that a raster uses the SOLWEIG land-cover codes, and
conversion of WRI OpenUrban land cover to them.

SOLWEIG codes (landcoverclasses_2016a.txt): 1 paved, 2 buildings, 5 grass,
6 bare soil, 7 water. compute_utci treats 3 and 4 (trees) as grass and any other
value as bare soil; trees themselves come from the Tree DSM.

Free of torch so that the preprocessor can import it. Run as a script to convert
an OpenUrban raster:

    python -m solweig_gpu.landcover OpenUrban.tif Landcover.tif
"""
import argparse

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

# OpenUrban code -> SOLWEIG code, as in CIF's city_metrix.layers.open_urban.reclass_map
# (the mapping CTCM applies before running SOLWEIG).
OPENURBAN_TO_SOLWEIG = {
    110: 5,  # green space, other
    120: 1,  # built up, other
    130: 6,  # barren
    200: 5,  # public open space
    300: 7,  # water
    400: 1,  # parking
    500: 1,  # roads
    600: 2, 601: 2, 602: 2,  # buildings, unclassified
    610: 2, 611: 2, 612: 2,  # buildings, residential
    620: 2, 621: 2, 622: 2,  # buildings, non-residential
}

SOLWEIG_CLASS_NAMES = {0: "nodata/unknown", 1: "paved", 2: "buildings", 5: "grass", 6: "bare soil",
                       7: "water"}

# Rows per block when converting (a 35000-pixel-wide block is ~70 MB as float64).
_BLOCK_ROWS = 256
# Evenly spaced rows read by check_landcover_codes.
_SAMPLE_ROWS = 64


def _valid(values, nodata):
    valid = np.isfinite(values)
    if nodata is not None and not np.isnan(nodata):
        valid &= values != nodata
    return valid


def check_landcover_codes(path):
    """
    Check that a land cover raster uses the SOLWEIG codes 1-7.

    compute_utci sets every other code to 6 (bare soil), so a raster in another
    scheme, e.g. OpenUrban (110-622), would otherwise run as bare soil everywhere.
    Evenly spaced rows are sampled; NaN and nodata pixels are ignored.

    Returns:
        float: Share of the sampled valid pixels outside 1-7

    Raises:
        ValueError: If there are no valid pixels, or more than half of them are outside 1-7
    """
    ds = gdal.Open(path)
    try:
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        rows = np.unique(np.linspace(0, ds.RasterYSize - 1, min(_SAMPLE_ROWS, ds.RasterYSize)).astype(int))
        values = np.concatenate([band.ReadAsArray(0, int(r), ds.RasterXSize, 1).ravel() for r in rows])
    finally:
        ds = None
    values = values.astype(np.float64)
    values = values[_valid(values, nodata)]
    if values.size == 0:
        raise ValueError(f"Land cover raster {path} has no valid pixels.")

    outside = float(np.mean((values < 1) | (values > 7)))
    if outside > 0.5:
        if np.isin(values, list(OPENURBAN_TO_SOLWEIG)).mean() > 0.5:
            hint = ("It looks like OpenUrban land cover: convert it first with "
                    f"`python -m solweig_gpu.landcover {path} Landcover.tif`.")
        else:
            hint = "Reclassify it to SOLWEIG codes first."
        raise ValueError(
            f"Land cover raster {path} must use the SOLWEIG codes 1-7 (1 paved, 2 buildings, 5 grass, "
            f"6 bare soil, 7 water), but {outside:.0%} of its pixels are outside that range and would "
            f"all run as bare soil. {hint}"
        )
    return outside


def openurban_to_solweig(src_path, dst_path):
    """
    Convert WRI OpenUrban land cover to SOLWEIG codes, on the same grid.

    Args:
        src_path (str): OpenUrban raster (codes 110-622)
        dst_path (str): Output GeoTIFF (Byte, tiled, DEFLATE) with SOLWEIG codes;
            NaN, nodata and codes not in OPENURBAN_TO_SOLWEIG become 0 (its nodata)

    Returns:
        dict: Pixel count per output code
    """
    lut = np.zeros(max(OPENURBAN_TO_SOLWEIG) + 1, dtype=np.uint8)
    for code, solweig in OPENURBAN_TO_SOLWEIG.items():
        lut[code] = solweig

    src = gdal.Open(src_path)
    dst = None
    try:
        band = src.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        width, height = src.RasterXSize, src.RasterYSize
        dst = gdal.GetDriverByName("GTiff").Create(
            dst_path, width, height, 1, gdal.GDT_Byte,
            options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
        )
        dst.SetGeoTransform(src.GetGeoTransform())
        dst.SetProjection(src.GetProjection())
        out_band = dst.GetRasterBand(1)
        out_band.SetNoDataValue(0)

        counts = np.zeros(lut.max() + 1, dtype=np.int64)
        for y in range(0, height, _BLOCK_ROWS):
            block = band.ReadAsArray(0, y, width, min(_BLOCK_ROWS, height - y)).astype(np.float64)
            known = _valid(block, nodata) & (block >= 0) & (block < lut.size) & (block == np.round(block))
            out = np.where(known, lut[np.where(known, block, 0).astype(np.int64)], 0).astype(np.uint8)
            out_band.WriteArray(out, 0, y)
            counts += np.bincount(out.ravel(), minlength=counts.size)
    finally:
        src = None
        dst = None  # flushes and closes the output

    return {code: int(n) for code, n in enumerate(counts) if n}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert WRI OpenUrban land cover to SOLWEIG land-cover codes.")
    parser.add_argument("src", help="OpenUrban raster (codes 110-622)")
    parser.add_argument("dst", help="Output GeoTIFF with SOLWEIG codes (Byte, nodata 0)")
    args = parser.parse_args(argv)

    counts = openurban_to_solweig(args.src, args.dst)
    total = sum(counts.values())
    for code, n in sorted(counts.items()):
        print(f"{code} {SOLWEIG_CLASS_NAMES.get(code, '?'):15s} {n / total:7.2%}")
    print(f"Wrote {args.dst}")


if __name__ == "__main__":
    main()
