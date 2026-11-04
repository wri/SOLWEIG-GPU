"""Download ERA-5 forcing data for SOLWEIG-GPU.

Two data sources are supported (``--source`` / ``source=``):

* ``era5`` (default) -- ERA-5 *single-levels* reanalysis at 0.25 deg (~31 km).
* ``era5-land``      -- ERA5-Land at 0.1 deg (~9 km): higher resolution, land
  only. See https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land .

Either way this produces the two NetCDF files that
``solweig_gpu.preprocessor.process_era5_data`` expects to find in the data
folder:

    data_stream-oper_stepType-instant.nc   (t2m, d2m, sp, u10, v10)
    data_stream-oper_stepType-accum.nc     (ssrd, strd)

The download area is taken automatically from your Building DSM / DEM raster
(so it always matches the city/domain you are modelling), and the date defaults
to the latest release (both streams run ~5 days behind real time). By default
all 24 hours of the day are fetched; pass ``--time`` / ``times`` to restrict the
download to specific hours of the day.

ERA5-Land note: its ssrd/strd are accumulated from 00 UTC and reset every day,
whereas single-levels (and the SOLWEIG preprocessor's ``ssrd / 3600``) expect
each step to hold only that one hour. This helper therefore de-accumulates the
ERA5-Land radiation into per-hour totals before writing ``...-accum.nc``, so the
two sources are interchangeable downstream. Because de-accumulation needs a
contiguous day, the ``era5-land`` source always downloads all 24 hours.

Prerequisites
-------------
* A free Copernicus Climate Data Store (CDS) account, with the licence for the
  dataset you use accepted on its page (``reanalysis-era5-single-levels`` for
  ``era5``; ``reanalysis-era5-land`` for ``era5-land``).
* A ``~/.cdsapirc`` file::

      url: https://cds.climate.copernicus.eu/api
      key: <YOUR-PERSONAL-ACCESS-TOKEN>

* ``pip install cdsapi``

See ``docs/input_data.md`` for the full walkthrough.

Usage
-----
Command line::

    python -m solweig_gpu.download_era5 --dsm /path/Building_DSM.tif \
        --out data/ERA-5_Forcing_data

    # higher-resolution ERA5-Land instead of single-levels:
    python -m solweig_gpu.download_era5 --dsm /path/Building_DSM.tif \
        --out data/ERA-5_Forcing_data --source era5-land

    # a specific date and only certain hours of the day (UTC):
    python -m solweig_gpu.download_era5 --dsm /path/Building_DSM.tif \
        --out data/ERA-5_Forcing_data --date 2024-08-13 --time 6 12 18

Python::

    from solweig_gpu.download_era5 import download_era5
    used_date = download_era5("Building_DSM.tif", "data/ERA-5_Forcing_data",
                              date="2024-08-13", times=[6, 12, 18])
    # ...or ERA5-Land (0.1 deg):
    used_date = download_era5("Building_DSM.tif", "data/ERA-5_Forcing_data",
                              source="era5-land")
"""

import argparse
import datetime as dt
import math
import os
import zipfile

from osgeo import gdal, osr

# CDS long-name -> resulting NetCDF short-name, grouped by step type.
# Instantaneous variables land in the *-instant.nc file, accumulated ones in
# the *-accum.nc file (this is what creates the two-file split SOLWEIG reads).
INSTANT_VARIABLES = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "surface_pressure": "sp",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
}
ACCUM_VARIABLES = {
    "surface_solar_radiation_downwards": "ssrd",
    "surface_thermal_radiation_downwards": "strd",
}
DEFAULT_VARIABLES = list(INSTANT_VARIABLES) + list(ACCUM_VARIABLES)

INSTANT_FILE = "data_stream-oper_stepType-instant.nc"
ACCUM_FILE = "data_stream-oper_stepType-accum.nc"

ERA5_GRID_DEG = 0.25  # ERA-5 single-levels horizontal resolution

# CDS dataset id and native grid spacing (degrees) per supported source.
DATASETS = {
    "era5": "reanalysis-era5-single-levels",
    "era5-land": "reanalysis-era5-land",
}
SOURCE_GRID_DEG = {"era5": 0.25, "era5-land": 0.1}


def latest_era5_date(buffer_days: int = 6) -> dt.date:
    """Return the most recent date for which ERA5T is reliably available.

    The ERA5T preliminary stream is published ~5 days behind real time; we use
    a 6-day buffer by default so a freshly-published (possibly partial) day is
    not requested.
    """
    return dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=buffer_days)


def domain_bbox_from_raster(
    raster_path: str, margin_deg: float = 0.5, grid_deg: float = ERA5_GRID_DEG
):
    """Return an ERA-5 ``area=[North, West, South, East]`` (degrees) covering the raster.

    The raster's four corners are reprojected to EPSG:4326, padded by
    ``margin_deg`` and snapped outward to the ``grid_deg`` grid (0.25 deg for
    single-levels, 0.1 deg for ERA5-Land) so at least one grid cell is always
    returned.
    """
    ds = gdal.Open(raster_path)
    if ds is None:
        raise FileNotFoundError(raster_path)
    proj = ds.GetProjection()
    if not proj:
        raise ValueError(
            f"{raster_path} has no CRS, so its lat/lon bounding box cannot be "
            "determined. Provide a georeferenced DSM/DEM raster."
        )

    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    corners = [(0, 0), (nx, 0), (0, ny), (nx, ny)]
    xs = [gt[0] + px * gt[1] + py * gt[2] for px, py in corners]
    ys = [gt[3] + px * gt[4] + py * gt[5] for px, py in corners]

    src = osr.SpatialReference()
    src.ImportFromWkt(proj)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    # Force (lon, lat) ordering for both CRSs regardless of GDAL version.
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)
    pts = [ct.TransformPoint(x, y)[:2] for x, y in zip(xs, ys)]  # (lon, lat)
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]

    q = grid_deg
    north = min(90.0, math.ceil((max(lats) + margin_deg) / q) * q)
    south = max(-90.0, math.floor((min(lats) - margin_deg) / q) * q)
    west = max(-180.0, math.floor((min(lons) - margin_deg) / q) * q)
    east = min(180.0, math.ceil((max(lons) + margin_deg) / q) * q)
    return [north, west, south, east]


def _rename_split_files(out_folder: str):
    """Give the extracted NetCDF members the names SOLWEIG expects.

    The CDS normally names them exactly ``data_stream-oper_stepType-instant.nc``
    / ``...-accum.nc``; this is a safety net that detects the instant vs accum
    file by its variables and renames if the naming ever differs.
    """
    import netCDF4

    for fn in os.listdir(out_folder):
        if not fn.endswith(".nc") or fn in (INSTANT_FILE, ACCUM_FILE):
            continue
        path = os.path.join(out_folder, fn)
        try:
            with netCDF4.Dataset(path) as ncf:
                variables = set(ncf.variables)
        except Exception:
            continue
        if "t2m" in variables:
            os.replace(path, os.path.join(out_folder, INSTANT_FILE))
        elif "ssrd" in variables:
            os.replace(path, os.path.join(out_folder, ACCUM_FILE))


def _ensure_canonical_files(out_folder: str):
    """Guarantee ``INSTANT_FILE`` and ``ACCUM_FILE`` exist in ``out_folder``.

    First applies :func:`_rename_split_files` (the normal case, where the CDS
    delivered one file per step type). As a fallback -- e.g. if a source ever
    returns a single combined NetCDF -- a lone file holding both an
    instantaneous (``t2m``) and an accumulated (``ssrd``) variable is split into
    the two expected files.
    """
    _rename_split_files(out_folder)
    have_instant = os.path.exists(os.path.join(out_folder, INSTANT_FILE))
    have_accum = os.path.exists(os.path.join(out_folder, ACCUM_FILE))
    if have_instant and have_accum:
        return

    import xarray as xr

    accum_short = set(ACCUM_VARIABLES.values())  # {'ssrd', 'strd'}
    instant_path = os.path.join(out_folder, INSTANT_FILE)
    accum_path = os.path.join(out_folder, ACCUM_FILE)
    for fn in os.listdir(out_folder):
        if not fn.endswith(".nc"):
            continue
        path = os.path.join(out_folder, fn)
        try:
            ds = xr.open_dataset(path, decode_times=True)
        except Exception:
            continue
        names = set(ds.data_vars)
        if "t2m" not in names or "ssrd" not in names:
            ds.close()
            continue
        instant_vars = [v for v in ds.data_vars if v not in accum_short]
        accum_vars = [v for v in ds.data_vars if v in accum_short]
        ds_instant = ds[instant_vars].load()
        ds_accum = ds[accum_vars].load()
        ds.close()
        # Write to temps first: the combined file may already have been renamed
        # to INSTANT_FILE by _rename_split_files, so we must not read/remove the
        # path we are about to write.
        ds_instant.to_netcdf(instant_path + ".tmp")
        ds_accum.to_netcdf(accum_path + ".tmp")
        if os.path.abspath(path) not in (
            os.path.abspath(instant_path),
            os.path.abspath(accum_path),
        ):
            os.remove(path)
        os.replace(instant_path + ".tmp", instant_path)
        os.replace(accum_path + ".tmp", accum_path)
        break


def _deaccumulate_era5_land(accum_path: str):
    """Rewrite ERA5-Land accumulations in ``accum_path`` as per-hour totals.

    ERA5-Land accumulates ssrd/strd from 00 UTC and resets every day, whereas
    ERA-5 single-levels (and the SOLWEIG preprocessor's ``ssrd / 3600``) expect
    each step to hold only that one hour's accumulation. We difference
    consecutive hourly steps so the file matches the single-levels convention
    (J m-2 per hour):

    * 02..23 UTC -> value(t) - value(t-1).
    * 01 UTC     -> the raw value already *is* the 00->01 accumulation.
    * 00 UTC     -> holds the previous day's 24 h total, which cannot be
      de-accumulated from a single day, so it is filled from the 01 UTC step
      (night, where downward shortwave is ~0 anyway).

    Requires contiguous hourly data, which is why the era5-land download always
    fetches all 24 hours.
    """
    import xarray as xr

    ds = xr.open_dataset(accum_path, decode_times=True)
    ds.load()
    ds.close()

    tname = (
        "valid_time"
        if ("valid_time" in ds.coords or "valid_time" in ds.dims)
        else "time"
    )
    if tname not in ds.coords and tname not in ds.dims:
        raise KeyError(
            f"{accum_path}: no 'valid_time'/'time' coordinate; cannot de-accumulate."
        )
    hour = ds[tname].dt.hour

    for name in list(ds.data_vars):
        da = ds[name]
        if tname not in da.dims:
            continue
        hourly = da - da.shift({tname: 1})  # per-hour increment
        hourly = xr.where(hour == 1, da, hourly)  # 01 UTC: raw == 00->01 accum
        hourly = xr.where(
            hour == 0, hourly.shift({tname: -1}), hourly
        )  # 00 UTC: persist next hour
        result = (
            hourly.clip(min=0.0).fillna(0.0).astype("float32")
        )  # downward flux is >= 0
        result.attrs = dict(da.attrs)
        ds[name] = result
    ds.attrs["solweig_deaccumulated"] = (
        "ssrd/strd differenced to per-hour totals from ERA5-Land daily accumulation"
    )

    tmp = accum_path + ".tmp"
    ds.to_netcdf(tmp)
    ds.close()
    os.replace(tmp, accum_path)


def _normalize_times(times):
    """Normalise requested hours into sorted, unique CDS ``"HH:00"`` strings.

    ERA-5 single-levels *reanalysis* is hourly on the hour, so each entry must
    resolve to a whole hour 0-23 (minutes must be ``00``). Accepts a list of
    ints/strings, or a single comma/space-separated string, in any of these
    forms: ``6``, ``"06"``, ``"6:00"``, ``"06:00"``, ``"06:00:00"``.
    ``None`` (the default) returns all 24 hours.
    """
    if times is None:
        return [f"{h:02d}:00" for h in range(24)]
    # Allow a single "6,12,18" or "6 12 18" string (handy from the shell/Python).
    if isinstance(times, (str, int)):
        times = str(times).replace(",", " ").split()
    hours = []
    for tok in times:
        s = str(tok).strip()
        if not s:
            continue
        parts = s.split(":")  # accept HH, HH:MM, HH:MM:SS
        try:
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
        except ValueError:
            raise ValueError(f"Invalid time {tok!r}; use an hour 0-23 or 'HH:00'.")
        if minute != 0:
            raise ValueError(
                f"Invalid time {tok!r}: ERA-5 reanalysis is hourly, so minutes "
                "must be 00."
            )
        if not 0 <= hour <= 23:
            raise ValueError(f"Invalid hour {hour} in {tok!r}; must be 0-23.")
        hours.append(hour)
    if not hours:
        raise ValueError("No valid times were provided.")
    return [f"{h:02d}:00" for h in sorted(set(hours))]


def download_era5(
    dsm_path,
    out_folder,
    date=None,
    times=None,
    margin_deg=0.5,
    variables=None,
    retry_back_days=3,
    source="era5",
):
    """Download one day of hourly ERA-5 covering ``dsm_path`` into ``out_folder``.

    Parameters
    ----------
    dsm_path : str
        Building DSM / DEM GeoTIFF that defines the area ("the input city").
    out_folder : str
        Destination folder; pass this as ``data_folder`` to ``thermal_comfort``.
    date : str | datetime.date | None
        ``YYYY-MM-DD`` to download; defaults to the latest available ERA5T date.
    times : list[str | int] | str | None
        Hours of the day to download, in UTC. Accepts e.g. ``[6, 12, 18]``,
        ``["06:00", "18:00"]`` or ``"6,12,18"``. Defaults to all 24 hours.
    margin_deg : float
        Degrees of padding added around the raster extent.
    variables : list[str] | None
        CDS long-form variable names; defaults to the set SOLWEIG needs.
    retry_back_days : int
        If the requested day is not yet published, step back up to this many days.
    source : str
        ``"era5"`` (default) for ERA-5 single-levels (0.25 deg), or
        ``"era5-land"`` for ERA5-Land (0.1 deg). For ``"era5-land"`` all 24
        hours are always fetched and ssrd/strd are de-accumulated to per-hour
        totals, so ``times`` is ignored for that source.

    Returns
    -------
    datetime.date
        The date that was actually downloaded.
    """
    import cdsapi

    if source not in DATASETS:
        raise ValueError(
            f"Unknown source {source!r}; choose one of {sorted(DATASETS)}."
        )

    os.makedirs(out_folder, exist_ok=True)
    variables = variables or DEFAULT_VARIABLES
    times = _normalize_times(times)
    # ERA5-Land radiation is de-accumulated over the whole day, so it needs a
    # contiguous 24 h regardless of any requested subset.
    if source == "era5-land":
        full_day = _normalize_times(None)
        if times != full_day:
            print(
                "  [era5-land] ssrd/strd are de-accumulated over the full day; "
                "fetching all 24 hours (ignoring the requested time subset)."
            )
        times = full_day
    area = domain_bbox_from_raster(dsm_path, margin_deg, SOURCE_GRID_DEG[source])
    if date is None:
        date = latest_era5_date()
    elif isinstance(date, str):
        date = dt.date.fromisoformat(date)

    dataset = DATASETS[source]
    client = cdsapi.Client()
    zip_path = os.path.join(out_folder, "_era5_download.zip")
    last_err = None
    used_date = None
    for back in range(retry_back_days + 1):
        d = date - dt.timedelta(days=back)
        request = {
            "variable": variables,
            "year": f"{d:%Y}",
            "month": f"{d:%m}",
            "day": f"{d:%d}",
            "time": times,
            "area": area,  # [North, West, South, East]
            "data_format": "netcdf",
            "download_format": "zip",
        }
        if source == "era5":
            request["product_type"] = "reanalysis"  # ERA5-Land has no product_type
        try:
            client.retrieve(dataset, request, zip_path)
            used_date = d
            break
        except Exception as exc:  # most recent day may not be published yet
            last_err = exc
            print(f"  {d} not available ({exc}); stepping back a day...")
    if used_date is None:
        raise RuntimeError(
            f"No {source} data found in the {retry_back_days + 1} days ending {date}: "
            f"{last_err}"
        )

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_folder)
    os.remove(zip_path)
    _ensure_canonical_files(out_folder)
    if source == "era5-land":
        # Convert daily-cumulative ssrd/strd to per-hour totals (single-levels
        # convention) so the preprocessor's `ssrd / 3600` stays correct.
        _deaccumulate_era5_land(os.path.join(out_folder, ACCUM_FILE))

    for fname in (INSTANT_FILE, ACCUM_FILE):
        if not os.path.exists(os.path.join(out_folder, fname)):
            raise RuntimeError(
                f"Expected {fname} was not produced. Check the requested variables."
            )
    print(f"{source} for {used_date}  area[N,W,S,E]={area}  ->  {out_folder}")
    return used_date


def main():
    parser = argparse.ArgumentParser(
        description="Download ERA-5 forcing data for a SOLWEIG-GPU run."
    )
    parser.add_argument(
        "--dsm",
        required=True,
        help="Building DSM / DEM GeoTIFF that defines the city/area to download for.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output folder; use this as data_folder in thermal_comfort().",
    )
    parser.add_argument(
        "--source",
        choices=sorted(DATASETS),
        default="era5",
        help="CDS dataset to download: 'era5' (single-levels, 0.25 deg; default) "
        "or 'era5-land' (0.1 deg, higher resolution; fetches all 24 hours).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date YYYY-MM-DD (default: latest available ERA5T, ~today minus 6 days).",
    )
    parser.add_argument(
        "--time",
        "--times",
        dest="times",
        nargs="+",
        default=None,
        metavar="HH",
        help="Hours of the day to download in UTC, e.g. --time 6 12 18 or "
        "--time 06:00 18:00 (default: all 24 hours).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Degrees of padding around the raster extent (default: 0.5).",
    )
    args = parser.parse_args()

    used = download_era5(
        args.dsm,
        args.out,
        date=args.date,
        times=args.times,
        margin_deg=args.margin,
        source=args.source,
    )
    # Reflect the actual hours fetched so the suggested window covers them
    # (era5-land always fetches all 24 hours).
    times = _normalize_times(None if args.source == "era5-land" else args.times)
    print(
        "\nNext, run the model for this date:\n"
        f"  thermal_comfort(..., data_source_type='ERA5', data_folder='{args.out}',\n"
        f"      selected_date_str='{used}',\n"
        f"      start_time='{used} {times[0]}:00', end_time='{used} {times[-1]}:00')"
    )


if __name__ == "__main__":
    main()
