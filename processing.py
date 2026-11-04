"""
SOLWEIG-GPU pipeline for one city and one date. run.py calls it with the
settings of a config file (config.toml): load_config(), then run_pipeline().

Split pipeline so repeat runs for a different time reuse the expensive,
geometry-only caches instead of recomputing them:

  download_era5     -> era5_folder/      (skips if present)      [time-specific]
  preprocess        -> re-tiles + (re)builds metfiles           [time-specific]
  run_walls_aspect  -> walls/ + aspect/  (skips if present)     [geometry-only, cached]
  calculate_svf     -> SVF/              (skips if present)      [geometry-only, cached]
  run_utci_tiles    -> output_folder/    (the SOLWEIG/UTCI run)  [time-specific]
  mosaic_tiles      -> output_folder/mosaic/

walls/aspect and SVF live in processed_inputs/{walls,aspect,SVF}, which the
tiling step does NOT touch, so they survive across runs. They are only valid
while the inputs, tile_size and overlap are unchanged. If you change the input
rasters, tile_size or overlap, delete processed_inputs/{walls,aspect,SVF}
first (or set overwrite_geometry = true in the config), otherwise the cached
tiles won't match the new tile geometry.
"""
import datetime as dt
import os
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10: pip install tomli
    import tomli as tomllib

import numpy as np
import pandas as pd
import xarray as xr

from solweig_gpu import (
    calculate_svf,
    mosaic_tiles,
    preprocess,
    run_utci_tiles,
    run_walls_aspect,
)
from solweig_gpu.download_era5 import (
    ACCUM_FILE,
    DATASETS,
    INSTANT_FILE,
    SOURCE_GRID_DEG,
    domain_bbox_from_raster,
    download_era5,
)
from solweig_gpu.landcover import openurban_to_solweig

# Every setting of the config file: [section] -> {key: allowed type(s)}.
SETTINGS = {
    "paths": {
        "base_path": str,
        "building_dsm": str,
        "dem": str,
        "trees": str,
        "openurban": str,
        "landcover": str,
        "albedo": str,
        "era5_folder": str,
    },
    # TOML reads unquoted dates and times as date/datetime objects.
    "time": {"date": (str, dt.date), "start": (str, dt.date), "end": (str, dt.date), "hours": list},
    "era5": {"source": str},
    "tiling": {"tile_size": int, "overlap": int, "overwrite_geometry": bool},
    "model": {"tree_transmissivity_pct": (int, float)},
    "outputs": {
        "save_tmrt": bool,
        "save_svf": bool,
        "save_kup": bool,
        "save_kdown": bool,
        "save_lup": bool,
        "save_ldown": bool,
        "save_shadow": bool,
        "clip_buildings": bool,
        "overwrite": bool,
        "split_bands": bool,
    },
}
# Settings that may be left out of the config file; they are then off (None).
OPTIONAL = {
    ("paths", "openurban"),
    ("paths", "landcover"),
    ("paths", "albedo"),
    ("time", "hours"),
    ("model", "tree_transmissivity_pct"),
}


def check_config(config):
    """Raise ValueError listing every unknown, missing or mistyped setting."""
    problems = [f"unknown section [{section}]" for section in sorted(config.keys() - SETTINGS.keys())]
    for section, types in SETTINGS.items():
        values = config.get(section, {})
        if not isinstance(values, dict):
            problems.append(f"[{section}] must be a section, not a single setting")
            continue
        problems += [f"unknown setting [{section}] {key}" for key in sorted(values.keys() - types.keys())]
        for key, allowed in types.items():
            if key not in values:
                if (section, key) not in OPTIONAL:
                    problems.append(f"missing setting [{section}] {key}")
            elif not isinstance(values[key], allowed):
                kinds = allowed if isinstance(allowed, tuple) else (allowed,)
                problems.append(f"[{section}] {key} = {values[key]!r} should be "
                                + " or ".join(kind.__name__ for kind in kinds))
    if not problems:
        if config["era5"]["source"] not in DATASETS:
            problems.append(f"[era5] source = {config['era5']['source']!r} should be one of {sorted(DATASETS)}")
        if config["paths"].get("openurban") and not config["paths"].get("landcover"):
            problems.append("[paths] openurban needs landcover, the file the converted land cover is written to")
    if problems:
        raise ValueError("Problems in the config file:\n  " + "\n  ".join(problems))


def load_config(path):
    """
    Read a TOML config file (see config.toml) and check it.

    A relative base_path is taken relative to the config file's folder.
    """
    with open(path, "rb") as f:
        config = tomllib.load(f)
    check_config(config)
    folder = os.path.dirname(os.path.abspath(path))
    config["paths"]["base_path"] = os.path.join(folder, os.path.expanduser(config["paths"]["base_path"]))
    return config


def _text(value, fmt):
    """A date/time setting as text: TOML reads unquoted ones as date/datetime objects (UTC is assumed)."""
    if isinstance(value, dt.datetime) and value.tzinfo is not None:
        value = value.astimezone(dt.timezone.utc)
    return value.strftime(fmt) if isinstance(value, dt.date) else value


def era5_ready(folder, dsm_path, start_time, end_time, source="era5"):
    """
    True if folder already holds ERA5 files of this source (see download_era5)
    that cover the DSM at every hour from start_time to end_time (UTC), so no
    download is needed.
    """
    grid = SOURCE_GRID_DEG[source]
    north, west, south, east = domain_bbox_from_raster(dsm_path, margin_deg=0.0, grid_deg=grid)
    hours = pd.date_range(start_time, end_time, freq="h")
    for name in (INSTANT_FILE, ACCUM_FILE):
        try:
            with xr.open_dataset(os.path.join(folder, name)) as ds:
                times = ds["valid_time" if "valid_time" in ds.coords else "time"].values
                lat, lon = ds["latitude"].values, ds["longitude"].values
        except (OSError, KeyError, ValueError):
            return False
        if not (
            hours.isin(times).all()
            and lat.min() <= south and lat.max() >= north
            and lon.min() <= west and lon.max() >= east
            # tells the 0.25° single-levels grid from ERA5-Land's 0.1°
            and np.allclose(np.abs(np.diff(lat)), grid)
        ):
            return False
    return True


def run_pipeline(config):
    """Run every step of the pipeline with the settings from load_config()."""
    paths, time, tiling, outputs = (config[key] for key in ("paths", "time", "tiling", "outputs"))
    base_path = paths["base_path"]

    def path(key):
        """A [paths] setting resolved against base_path."""
        return os.path.join(base_path, os.path.expanduser(paths[key]))

    def optional_path(key):
        """An optional [paths] setting resolved against base_path, or None if it is left out."""
        return path(key) if paths.get(key) else None

    selected_date_str = _text(time["date"], "%Y-%m-%d")
    start_time = _text(time["start"], "%Y-%m-%d %H:%M:%S")
    end_time = _text(time["end"], "%Y-%m-%d %H:%M:%S")
    source = config["era5"]["source"]
    era5_folder = path("era5_folder")
    dsm_path = path("building_dsm")

    # 0) Time-specific: ERA5 forcing (all 24 h of the date) from the Copernicus CDS,
    #    over the Building DSM's area plus 0.5°. Needs a CDS account and ~/.cdsapirc
    #    (see solweig_gpu/download_era5.py). Skipped when era5_folder already
    #    covers the window, so reruns don't queue at the CDS.
    if era5_ready(era5_folder, dsm_path, start_time, end_time, source):
        print(f"[skip] ERA5 in {era5_folder} already covers {start_time} - {end_time} UTC")
    else:
        # retry_back_days=0: fail instead of quietly fetching an earlier day.
        download_era5(dsm_path, era5_folder, date=selected_date_str, source=source, retry_back_days=0)
        if not era5_ready(era5_folder, dsm_path, start_time, end_time, source):
            raise RuntimeError(
                f"The ERA5 download for {selected_date_str} does not cover "
                f"{start_time} - {end_time} UTC; keep that window within {selected_date_str}."
            )

    if optional_path("openurban"):
        openurban_to_solweig(src_path=path("openurban"), dst_path=path("landcover"))

    # 1) Time-specific: rebuild tiles for this run and (re)generate the metfiles
    #    for the date. Cheap relative to walls/aspect/SVF.
    preprocess_dir = preprocess(
        base_path=base_path,
        selected_date_str=selected_date_str,
        building_dsm_filename=dsm_path,
        dem_filename=path("dem"),
        trees_filename=path("trees"),
        tile_size=tiling["tile_size"],
        overlap=tiling["overlap"],
        landcover_filename=optional_path("landcover"),
        albedo_filename=optional_path("albedo"),
        use_own_met=False,
        data_source_type="ERA5",
        data_folder=era5_folder,
        start_time=start_time,
        end_time=end_time,
        hours=time.get("hours"),
    )

    # 2) Geometry-only: computed once, then reused. Both skip tiles that already
    #    exist, so on later runs these return almost immediately, unless
    #    overwrite_geometry is set (after the geometry / tile_size / overlap change).
    run_walls_aspect(preprocess_dir, overwrite=tiling["overwrite_geometry"])
    calculate_svf(preprocess_dir, overwrite=tiling["overwrite_geometry"])

    # 3) Time-specific: the actual SOLWEIG/UTCI run for this window. Unless
    #    overwrite is set, it skips any tile whose requested outputs already exist
    #    in output_folder/ (which preprocess never touches), so a run that was
    #    interrupted part-way resumes instead of recomputing finished tiles.
    run_utci_tiles(
        base_path=base_path,
        preprocess_dir=preprocess_dir,
        selected_date_str=selected_date_str,
        save_tmrt=outputs["save_tmrt"],
        save_svf=outputs["save_svf"],
        save_kup=outputs["save_kup"],
        save_kdown=outputs["save_kdown"],
        save_lup=outputs["save_lup"],
        save_ldown=outputs["save_ldown"],
        save_shadow=outputs["save_shadow"],
        clip_buildings=outputs["clip_buildings"],
        tree_transmissivity_pct=config.get("model", {}).get("tree_transmissivity_pct"),
        overwrite=outputs["overwrite"],
    )

    # 4) Post-process: stitch the per-tile GeoTIFFs in output_folder/{i_j}/ into
    #    combined rasters under output_folder/mosaic/. Each tile keeps only its
    #    tile_size core, so the overlap fringe is dropped and the mosaic matches
    #    the original raster grid.
    mosaic_dir = mosaic_tiles(base_path=base_path, tile_size=tiling["tile_size"], split_bands=outputs["split_bands"])
    print(f"Mosaics written to: {mosaic_dir}")
