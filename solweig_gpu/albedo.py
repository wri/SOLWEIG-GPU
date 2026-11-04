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
Albedo inputs: albedo raster checks and grid alignment, the cool-roof
air-temperature cut, and the ALBEDO tag that lets resumed runs detect outputs
computed with other albedo settings.

Free of torch so that both the preprocessor and utci_process can import it.
"""
import hashlib
import json
import os

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

# GeoTIFF metadata item on the UTCI output recording the albedo settings it was
# computed with (see albedo_tag).
ALBEDO_TAG = "ALBEDO"

WALL_ALBEDO_DEFAULT = 0.2
GROUND_ALBEDO_DEFAULT = 0.15

# Krayenhoff et al. (2021), doi:10.1088/1748-9326/abdcf1: a 0.1 increase in
# albedo lowers midday clear-sky air temperature by 0.6 °C.
KRAYENHOFF_COEFF = 6.0
# (local hour, fraction of the 12:00 cut) as used by cities-heat-resilient-infrastructure.
# The cut is interpolated between these hours and not applied outside them.
AIR_TEMP_CUT_PROFILE = ((12.0, 1.0), (15.0, 0.935315), (18.0, 0.646853))

AIR_TEMP_CUT_FILE = "air_temp_cut.json"

# Slack for albedo values that bilinear resampling nudges past 0 or 1.
_RANGE_TOL = 1e-6


def check_fraction(name, value):
    """Return value as a float, raising ValueError unless it lies in [0, 1]."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


def air_temp_cut_factor(local_hours):
    """
    Fraction of the 12:00 air-temperature cut to apply at each local time.

    Args:
        local_hours (array-like): Local time as decimal hours (e.g. 14.5 for 14:30)

    Returns:
        np.ndarray: Factors, interpolated along AIR_TEMP_CUT_PROFILE and 0 outside it
    """
    hours = np.asarray(local_hours, dtype=float)
    xs, ys = zip(*AIR_TEMP_CUT_PROFILE)
    inside = (hours >= xs[0]) & (hours <= xs[-1])
    return np.where(inside, np.interp(hours, xs, ys), 0.0)


def check_albedo_bounds(lo, hi, path):
    """Raise ValueError unless an albedo raster's value range lo..hi lies within [0, 1]."""
    if lo < -_RANGE_TOL or hi > 1 + _RANGE_TOL:
        raise ValueError(
            f"Albedo raster {path} must hold fractions between 0 and 1, but its values range "
            f"from {lo:g} to {hi:g}. If it is in percent (0-100), divide it by 100."
        )


def check_albedo_range(path):
    """
    Check that an albedo raster holds fractions between 0 and 1.

    NaN and nodata pixels are ignored.

    Returns:
        tuple: (min, max) of the valid pixels

    Raises:
        ValueError: If there are no valid pixels or any value is outside [0, 1]
    """
    ds = gdal.Open(path)
    try:
        lo, hi = ds.GetRasterBand(1).ComputeRasterMinMax(False)
    except RuntimeError as err:
        raise ValueError(f"Albedo raster {path} has no valid pixels.") from err
    finally:
        ds = None
    check_albedo_bounds(lo, hi, path)
    return lo, hi


def _grids_match(ds, ref):
    if (ds.RasterXSize, ds.RasterYSize) != (ref.RasterXSize, ref.RasterYSize):
        return False
    if not np.allclose(ds.GetGeoTransform(), ref.GetGeoTransform(), rtol=0, atol=1e-6):
        return False
    srs, ref_srs = ds.GetSpatialRef(), ref.GetSpatialRef()
    if srs is None or ref_srs is None:
        return srs is None and ref_srs is None
    return bool(srs.IsSame(ref_srs))


def align_to_grid(src_path, ref_path, dst_path):
    """
    Put a raster onto another raster's grid.

    Args:
        src_path (str): Raster to align (e.g. a 10 m albedo raster)
        ref_path (str): Raster whose grid to match (the Building DSM)
        dst_path (str): Where to write the resampled raster, as a VRT

    Returns:
        str: src_path if the grids already match, otherwise dst_path, a bilinear
            resampling of src_path in which pixels it does not cover are NaN
    """
    src = gdal.Open(src_path)
    ref = gdal.Open(ref_path)
    try:
        if _grids_match(src, ref):
            return src_path
        if src.GetSpatialRef() is None:
            raise ValueError(f"{src_path} has no CRS, so it cannot be resampled onto the grid of {ref_path}.")
        src_nodata = src.GetRasterBand(1).GetNoDataValue()
        gt = ref.GetGeoTransform()
        width, height = ref.RasterXSize, ref.RasterYSize
        out = gdal.Warp(
            dst_path,
            os.path.abspath(src_path),
            format="VRT",
            dstSRS=ref.GetProjection(),
            outputBounds=(gt[0], gt[3] + gt[5] * height, gt[0] + gt[1] * width, gt[3]),
            width=width,
            height=height,
            resampleAlg="bilinear",
            outputType=gdal.GDT_Float32,
            srcNodata=float("nan") if src_nodata is None else src_nodata,
            dstNodata=float("nan"),
        )
        out = None  # writes the VRT
    finally:
        src = None
        ref = None
    print(f"[INFO] Resampled {src_path} onto the grid of {ref_path}: {dst_path}")
    return dst_path


def _valid(arr, nodata):
    mask = np.isfinite(arr)
    if nodata is not None and not np.isnan(nodata):
        mask &= arr != nodata
    return mask


def mean_albedo_change(scenario_path, baseline_path, block_rows=512):
    """
    Mean of (scenario - baseline) albedo over the pixels valid in both rasters.

    Both rasters must share one grid (see align_to_grid). They are read in row
    blocks so that a city-sized domain does not have to fit in memory.
    """
    scen = gdal.Open(scenario_path)
    base = gdal.Open(baseline_path)
    try:
        if (scen.RasterXSize, scen.RasterYSize) != (base.RasterXSize, base.RasterYSize):
            raise ValueError("The scenario and baseline albedo rasters must share a grid.")
        scen_band, base_band = scen.GetRasterBand(1), base.GetRasterBand(1)
        scen_nodata, base_nodata = scen_band.GetNoDataValue(), base_band.GetNoDataValue()
        width, height = scen.RasterXSize, scen.RasterYSize
        total, count = 0.0, 0
        for row in range(0, height, block_rows):
            rows = min(block_rows, height - row)
            s = scen_band.ReadAsArray(0, row, width, rows).astype(np.float64)
            b = base_band.ReadAsArray(0, row, width, rows).astype(np.float64)
            valid = _valid(s, scen_nodata) & _valid(b, base_nodata)
            total += float((s[valid] - b[valid]).sum())
            count += int(valid.sum())
    finally:
        scen = None
        base = None
    if count == 0:
        raise ValueError("The scenario and baseline albedo rasters have no valid pixels in common.")
    return total / count


def write_air_temp_cut(albedo_dir, delta_albedo, albedo_path, baseline_path):
    """
    Record the cool-roof air-temperature cut next to the albedo tiles.

    Returns:
        float: The 12:00 cut in °C (KRAYENHOFF_COEFF * delta_albedo)
    """
    cut = KRAYENHOFF_COEFF * delta_albedo
    record = {
        "delta_albedo": delta_albedo,
        "air_temp_cut_noon_c": cut,
        "coefficient_c_per_unit_albedo": KRAYENHOFF_COEFF,
        "profile_local_hour_factor": [list(point) for point in AIR_TEMP_CUT_PROFILE],
        "albedo": os.path.abspath(albedo_path),
        "baseline_albedo": os.path.abspath(baseline_path),
    }
    os.makedirs(albedo_dir, exist_ok=True)
    with open(os.path.join(albedo_dir, AIR_TEMP_CUT_FILE), "w") as f:
        json.dump(record, f, indent=2)
    return cut


def read_air_temp_cut(albedo_dir):
    """Return the 12:00 air-temperature cut (°C) recorded by preprocess, or 0.0 if there is none."""
    path = os.path.join(albedo_dir, AIR_TEMP_CUT_FILE)
    if not os.path.isfile(path):
        return 0.0
    with open(path) as f:
        return float(json.load(f)["air_temp_cut_noon_c"])


def albedo_digest(path):
    """Short fingerprint of an albedo tile's pixels and georeferencing."""
    ds = gdal.Open(path)
    try:
        band = ds.GetRasterBand(1)
        digest = hashlib.sha1(band.ReadAsArray().astype(np.float32).tobytes())
        digest.update(repr((ds.RasterXSize, ds.RasterYSize, ds.GetGeoTransform(),
                            band.GetNoDataValue())).encode())
    finally:
        ds = None
    return digest.hexdigest()[:16]


def albedo_tag(albedo_path, albedo_ground, albedo_walls, air_temp_cut):
    """
    Value of the ALBEDO metadata item for outputs computed with these settings.

    Args:
        albedo_path (str): Albedo tile used for the ground, or None
        albedo_ground (float): Default ground albedo
        albedo_walls (float): Wall albedo
        air_temp_cut (float): 12:00 air-temperature cut in °C
    """
    raster = albedo_digest(albedo_path) if albedo_path else "none"
    # "+ 0.0" turns -0.0 into 0.0 so both give the same tag.
    return (f"raster={raster};ground={float(albedo_ground) + 0.0:.6g};"
            f"walls={float(albedo_walls) + 0.0:.6g};cut={float(air_temp_cut) + 0.0:.6g}")


# Outputs written before the tag existed always used the default albedo.
LEGACY_ALBEDO_TAG = albedo_tag(None, GROUND_ALBEDO_DEFAULT, WALL_ALBEDO_DEFAULT, 0.0)


def albedo_state(path):
    """
    Albedo settings recorded on an existing UTCI output.

    Args:
        path (str): Path to a UTCI GeoTIFF written by compute_utci

    Returns:
        str or None: The ALBEDO metadata item, LEGACY_ALBEDO_TAG for files that
            predate it, or None if the file cannot be read (callers should
            recompute it).
    """
    try:
        ds = gdal.Open(path)
    except RuntimeError:
        return None
    if ds is None:
        return None
    value = ds.GetMetadataItem(ALBEDO_TAG)
    ds = None
    return LEGACY_ALBEDO_TAG if value is None else value
