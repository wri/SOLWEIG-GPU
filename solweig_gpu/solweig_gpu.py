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
from typing import Optional, List

# Lazy imports used below to avoid loading heavy deps at import time.


from typing import Optional, List

def preprocess(
    base_path: str,
    selected_date_str: str,
    building_dsm_filename: str = 'Building_DSM.tif',
    dem_filename: str = 'DEM.tif',
    trees_filename: str = 'Trees.tif',
    landcover_filename: Optional[str] = None,
    windcoeff_filename: Optional[str] = None,
    tile_size: int = 3600,
    overlap: int = 20,
    use_own_met: bool = True,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    data_source_type: Optional[str] = None,
    data_folder: Optional[str] = None,
    own_met_file: Optional[str] = None,
    preprocess_dir: Optional[str] = None,
    use_uhi: bool = True,
    hours: Optional[List[int]] = None,
    albedo_filename: Optional[str] = None,
    baseline_albedo_filename: Optional[str] = None,
) -> str:
    """
    Run preprocessing only: validate rasters, create tiles, and prepare metfiles.

    Use this when you want to run preprocessing once and then call
    :func:`run_walls_aspect` and :func:`run_utci_tiles` separately.

    Args:
        base_path: Base directory; used to resolve relative raster paths.
        selected_date_str: Simulation date 'YYYY-MM-DD'.
        building_dsm_filename, dem_filename, trees_filename, landcover_filename,
        windcoeff_filename:
            Raster paths or filenames (relative to base_path or absolute).
        tile_size: Tile size in pixels.
        overlap: Overlap between tiles in pixels.
        use_own_met: If True, use own_met_file; else use ERA5/WRF.
        start_time, end_time: Required for ERA5/WRF (UTC 'YYYY-MM-DD HH:MM:SS').
        data_source_type: 'ERA5' or 'wrfout' when use_own_met is False.
        data_folder: Folder with ERA5/WRF NetCDF files when use_own_met is False.
        own_met_file: Path to custom met file when use_own_met is True.
        preprocess_dir: Directory for preprocessing outputs. Defaults to
            ``{base_path}/processed_inputs``.
        use_uhi: If True, use UHI-aware ERA5 processing and write UHI_CYCLE/uhii
            into generated metfiles when available. If False, use standard ERA5
            processing and write uhii = 0.0.
        hours: Optional list of UTC hours-of-day to simulate (e.g. [4, 9, 12]).
            If None, every timestep in [start_time, end_time] is used. The window
            must span the requested hours. Ignored when use_own_met is True.
            Each hour gets the same sun position as in a full-day run: the middle
            of the hour its radiation is averaged over (30 min before its label).
        albedo_filename: Optional ground albedo raster (fractions 0-1, e.g. CIF
            cloud-masked albedo). It replaces the land-cover or default ground
            albedo; any grid is resampled onto the Building DSM grid.
        baseline_albedo_filename: Optional baseline albedo raster for cool-roof
            scenarios (requires albedo_filename). The forcing air temperature is
            then lowered by 6 x (mean albedo increase) °C at 12:00 local time,
            scaled down to 18:00 and not applied outside 12:00-18:00.

    Returns:
        The path to the preprocessing directory (tiles and metfiles written there).
    """
    import os
    from .preprocessor import ppr

    if preprocess_dir is None:
        preprocess_dir = os.path.join(base_path, "processed_inputs")
    os.makedirs(preprocess_dir, exist_ok=True)

    ppr(
        base_path,
        building_dsm_filename,
        dem_filename,
        trees_filename,
        landcover_filename,
        windcoeff_filename,
        tile_size,
        overlap,
        selected_date_str,
        use_own_met,
        start_time,
        end_time,
        data_source_type,
        data_folder,
        own_met_file,
        preprocess_dir=preprocess_dir,
        use_uhi=use_uhi,
        hours=hours,
        albedo_filename=albedo_filename,
        baseline_albedo_filename=baseline_albedo_filename,
    )
    return preprocess_dir


def build_inputs(
    lat: float,
    lon: float,
    city: Optional[str] = None,
    km_buffer: float = 8.0,
    km_reduced_lat: float = 3.0,
    km_reduced_lon: float = 1.0,
    year_start: int = 2024,
    year_end: int = 2025,
    base_folder: Optional[str] = None,
    resolution: float = 2.0,
) -> str:
    """
    Build static and meteorological inputs for SOLWEIG-GPU at a given location.

    Downloads and processes WorldCover, tree DSM, DEM, LCZ, OSM/GBA vectors,
    builds Building_DSM, Trees, DEM, Landuse rasters and meteorological NetCDF,
    and computes wind coefficient. Requires optional dependencies (e.g. earthengine-api,
    geemap, geopandas, osmnx); install them if you use this step.

    Use the returned path as ``base_path`` for :func:`preprocess` (with raster
    filenames like ``Building_DSM.tif``, ``DEM.tif``, ``Trees.tif``, ``Landuse.tif``
    in that folder) and optionally set ``use_own_met=False`` with the generated
    NetCDF in ``data_folder``.

    Args:
        lat, lon: Center of the area (degrees).
        city: Name for the output folder. If None, derived from reverse geocoding.
        km_buffer: Half-size of initial bounding box in km.
        km_reduced_lat, km_reduced_lon: Shrink (N/S and E/W) from bbox for SOLWEIG in km.
        year_start, year_end: Start/end year for meteorology (inclusive).
        base_folder: Workspace root. Defaults to the create_inputs module default.
        resolution: Reference grid resolution in meters.

    Returns:
        Path to the output directory (e.g. ``{base_folder}/{city}_for_solweig``)
        containing Building_DSM.tif, DEM.tif, Trees.tif, Landuse.tif and met NetCDF.
    """
    from .create_inputs import run_create_inputs

    return run_create_inputs(
        lat=lat,
        lon=lon,
        city=city,
        km_buffer=km_buffer,
        km_reduced_lat=km_reduced_lat,
        km_reduced_lon=km_reduced_lon,
        year_start=year_start,
        year_end=year_end,
        base_folder=base_folder,
        resolution=resolution,
    )


def run_walls_aspect(preprocess_dir: str, overwrite: bool = False) -> None:
    """
    Run wall height and aspect calculation for all tiles in the preprocessing directory.

    Call this after :func:`preprocess`. Writes to ``{preprocess_dir}/walls`` and
    ``{preprocess_dir}/aspect``.

    Walls and aspect depend only on geometry, so by default existing tiles are
    reused across runs (e.g. when only the simulation time changes). Pass
    ``overwrite=True`` to force recomputation after changing the input rasters,
    ``tile_size`` or ``overlap``.

    Args:
        preprocess_dir: Path returned by :func:`preprocess` (contains
            Building_DSM/, DEM/, Trees/, etc.).
        overwrite: If True, recompute and overwrite existing walls/aspect tiles.
    """
    import os
    from .walls_aspect import run_parallel_processing

    building_dsm_dir = os.path.join(preprocess_dir, "Building_DSM")
    walls_dir = os.path.join(preprocess_dir, "walls")
    aspect_dir = os.path.join(preprocess_dir, "aspect")
    run_parallel_processing(building_dsm_dir, walls_dir, aspect_dir, overwrite=overwrite)

def calculate_svf(base_path: str, patch_option: int = 2, overwrite: bool = False) -> None:
    """
    Calculate standalone Sky View Factor outputs for all raster tiles in a
    preprocessing directory.
    """
    import os
    import re
    import torch
    from tqdm import tqdm

    from .shadow import svf_calculator

    building_dsm_dir = os.path.join(base_path, "Building_DSM")
    dem_dir = os.path.join(base_path, "DEM")
    trees_dir = os.path.join(base_path, "Trees")
    svf_output_dir = os.path.join(base_path, "SVF")

    os.makedirs(svf_output_dir, exist_ok=True)

    for required_dir in [building_dsm_dir, dem_dir, trees_dir]:
        if not os.path.isdir(required_dir):
            raise FileNotFoundError(f"Required directory not found: {required_dir}")

    def extract_tile_key(filename: str, prefix: str) -> str | None:
        """
        Extract tile key from filenames like:
            Building_DSM_0_0.tif
            DEM_0_0.tif
            Trees_0_0.tif
        """
        pattern = rf"^{re.escape(prefix)}_(.+)\.tif$"
        match = re.match(pattern, filename)
        if match:
            return match.group(1)
        return None

    def map_tiles(folder: str, prefix: str) -> dict[str, str]:
        tile_map = {}

        for filename in os.listdir(folder):
            if filename.startswith(".") or not filename.lower().endswith(".tif"):
                continue

            key = extract_tile_key(filename, prefix)

            if key is not None:
                tile_map[key] = os.path.join(folder, filename)

        return tile_map

    building_tiles = map_tiles(building_dsm_dir, "Building_DSM")
    dem_tiles = map_tiles(dem_dir, "DEM")
    tree_tiles = map_tiles(trees_dir, "Trees")

    common_keys = sorted(
        set(building_tiles.keys())
        & set(dem_tiles.keys())
        & set(tree_tiles.keys())
    )

    if not common_keys:
        raise RuntimeError(
            "No matching SVF tiles found across Building_DSM/, DEM/, and Trees/."
        )

    missing_dem = sorted(set(building_tiles.keys()) - set(dem_tiles.keys()))
    missing_trees = sorted(set(building_tiles.keys()) - set(tree_tiles.keys()))

    if missing_dem:
        print(f"[WARNING] {len(missing_dem)} Building_DSM tiles are missing matching DEM tiles.")

    if missing_trees:
        print(f"[WARNING] {len(missing_trees)} Building_DSM tiles are missing matching Trees tiles.")

    print(f"[INFO] Found {len(common_keys)} matching SVF tiles.")
    print(f"[INFO] Writing outputs to: {svf_output_dir}")

    for key in tqdm(common_keys, desc="Calculating standalone SVF", mininterval=1):
        expected_tif = os.path.join(svf_output_dir, f"SkyViewFactor_{key}.tif")
        expected_zip = os.path.join(svf_output_dir, f"svfs_{key}.zip")
        expected_npz = os.path.join(svf_output_dir, f"shadowmats_{key}.npz")

        if (
            not overwrite
            and os.path.exists(expected_tif)
            and os.path.exists(expected_zip)
            and os.path.exists(expected_npz)
        ):
            continue

        svf_results = svf_calculator(patch_option=patch_option, save_rasters=True, building_dsm_path=building_tiles[key],
            tree_path=tree_tiles[key], dem_path=dem_tiles[key], output_dir=svf_output_dir, number=key,)

        # Do not keep GPU tensors in memory after each tile. The caching allocator reuses these freed
        # blocks for the next tile; calling empty_cache() here would instead cudaFree everything and
        # force an expensive cudaMalloc (+ device sync) on the next tile, so it is intentionally omitted.
        del svf_results

    print("[INFO] Standalone SVF calculation complete.")

def run_utci_tiles(
    base_path: str,
    preprocess_dir: str,
    selected_date_str: str,
    tile_keys: Optional[List[str]] = None,
    save_tmrt: bool = True,
    save_svf: bool = False,
    save_kup: bool = False,
    save_kdown: bool = False,
    save_lup: bool = False,
    save_ldown: bool = False,
    save_shadow: bool = False,
    save_wbgt: bool = False,
    save_ta: bool = False,
    save_wind: bool = False,
    clip_buildings: bool = True,
    overwrite: bool = False,
    albedo_walls: float = 0.2,
    albedo_ground: float = 0.15,
    air_temp_cut: Optional[float] = None,
    tree_transmissivity_pct: Optional[float] = None,
) -> None:
    """
    Run UTCI (and optional outputs) for tiles in the preprocessing directory.

    Call this after :func:`preprocess` and :func:`run_walls_aspect`. Writes
    GeoTIFFs to ``{base_path}/output_folder/{tile_key}/``.

    Args:
        base_path: Base directory; output_folder is created under this.
        preprocess_dir: Path returned by :func:`preprocess`.
        selected_date_str: Simulation date 'YYYY-MM-DD'.
        tile_keys: If None, process all tiles. If a list, process only those tile keys.
        save_tmrt, save_svf, save_kup, save_kdown, save_lup, save_ldown, save_shadow:
            Which outputs to save (UTCI is always saved).
        save_ta: Save diagnostic Ta field.
        save_wind: Save diagnostic wind field.
        clip_buildings: If True (default), building footprints (Building DSM -
            DEM >= 2 m) are NaN in the UTCI and Shadow outputs. If False, those
            outputs keep the rooftop values.
        overwrite: If False (default), skip any tile whose UTCI output and every
            other requested output already exist on disk, so an interrupted run
            resumes instead of recomputing finished tiles. A tile whose UTCI or
            Shadow output was written with a different ``clip_buildings`` or
            ``tree_transmissivity_pct`` setting (or before the latter was
            recorded), or whose UTCI was written with different albedo settings,
            is recomputed. Pass True to force recomputation and overwrite
            existing tile outputs.
        albedo_walls: Wall albedo (0-1).
        albedo_ground: Ground albedo (0-1) used where there is neither land cover
            nor an albedo raster.
        air_temp_cut: Cool-roof air-temperature cut at 12:00 local time (°C). None
            (default) uses the cut that :func:`preprocess` derived from
            ``baseline_albedo_filename``, or no cut if there was none.
        tree_transmissivity_pct: Percent (0-100) of the light, direct sun and
            diffuse sky light, that gets through tree canopies on every date,
            e.g. 3 for trees in leaf all year as in tropical cities. None
            (default) uses a temperate deciduous season: 3% on days 98-299, and
            outside it 50% of the direct sun (3% of the diffuse light).
    """
    import os
    import numpy as np
    import torch
    from .albedo import air_temp_cut_factor, albedo_state, albedo_tag, check_fraction, read_air_temp_cut
    from .utci_process import (check_tree_transmissivity, clip_state, compute_utci, map_files_by_key,
                               map_metfiles_by_key, tree_state, tree_tag)

    albedo_walls = check_fraction("albedo_walls", albedo_walls)
    albedo_ground = check_fraction("albedo_ground", albedo_ground)
    tree_transmissivity_pct = check_tree_transmissivity(tree_transmissivity_pct)

    base_output_path = os.path.join(base_path, "output_folder")
    input_met = os.path.join(preprocess_dir, "metfiles")
    building_dsm_dir = os.path.join(preprocess_dir, "Building_DSM")
    tree_dir = os.path.join(preprocess_dir, "Trees")
    dem_dir = os.path.join(preprocess_dir, "DEM")
    landcover_dir = os.path.join(preprocess_dir, "Landcover")
    windcoeff_dir = os.path.join(preprocess_dir, "WindCoeff")
    albedo_dir = os.path.join(preprocess_dir, "Albedo")
    walls_dir = os.path.join(preprocess_dir, "walls")
    aspect_dir = os.path.join(preprocess_dir, "aspect")

    building_dsm_map = map_files_by_key(building_dsm_dir, ".tif")
    tree_map = map_files_by_key(tree_dir, ".tif")
    dem_map = map_files_by_key(dem_dir, ".tif")
    landcover_map = map_files_by_key(landcover_dir, ".tif") if os.path.isdir(landcover_dir) else {}
    windcoeff_map = map_files_by_key(windcoeff_dir, ".tif") if os.path.isdir(windcoeff_dir) else {}
    albedo_map = map_files_by_key(albedo_dir, ".tif") if os.path.isdir(albedo_dir) else {}
    walls_map = map_files_by_key(walls_dir, ".tif")
    aspect_map = map_files_by_key(aspect_dir, ".tif")
    met_map = map_metfiles_by_key(input_met, selected_date_str)

    common_keys = set(building_dsm_map) & set(tree_map) & set(dem_map) & set(met_map) & set(walls_map) & set(aspect_map)

    if landcover_map:
        common_keys &= set(landcover_map)

    if albedo_map:
        common_keys &= set(albedo_map)

    if air_temp_cut is None:
        air_temp_cut = read_air_temp_cut(albedo_dir)
    air_temp_cut = float(air_temp_cut)
    if air_temp_cut:
        print(f"[INFO] Cool-roof air-temperature cut: {air_temp_cut:.2f} °C at 12:00 local time")
    cut_hours_checked = False

    # Do NOT force intersection with windcoeff_map, because windcoeff is optional
    if tile_keys is not None:
        common_keys = common_keys & set(tile_keys)
        if not common_keys:
            raise ValueError(f"No tiles to run; tile_keys={tile_keys} not found in preprocess_dir")

    def _numeric_key(k: str):
        x, y = k.split("_")
        return (int(x), int(y))

    print("Running Solweig ...")
    for key in sorted(common_keys, key=_numeric_key):
        building_dsm_path = building_dsm_map[key]
        tree_path = tree_map[key]
        dem_path = dem_map[key]
        landcover_path = landcover_map.get(key) if landcover_map else None
        windcoeff_path = windcoeff_map.get(key) if windcoeff_map else None
        albedo_path = albedo_map.get(key) if albedo_map else None
        walls_path = walls_map[key]
        aspect_path = aspect_map[key]
        met_file_path = met_map[key]

        output_folder = os.path.join(base_output_path, key)

        if not overwrite:
            # UTCI is always written; each enabled save_* flag adds one GeoTIFF,
            # all written together at the end of compute_utci (so a tile is
            # effectively all-or-nothing). SVF is intentionally excluded: in the
            # normal pipeline the SVF cache exists, so compute_utci reuses it and
            # never writes SVF_{key}.tif here -- its absence is not "unfinished".
            expected = [os.path.join(output_folder, f"UTCI_{key}.tif")]
            if save_tmrt:
                expected.append(os.path.join(output_folder, f"TMRT_{key}.tif"))
            if save_kup:
                expected.append(os.path.join(output_folder, f"Kup_{key}.tif"))
            if save_kdown:
                expected.append(os.path.join(output_folder, f"Kdown_{key}.tif"))
            if save_lup:
                expected.append(os.path.join(output_folder, f"Lup_{key}.tif"))
            if save_ldown:
                expected.append(os.path.join(output_folder, f"Ldown_{key}.tif"))
            if save_shadow:
                expected.append(os.path.join(output_folder, f"Shadow_{key}.tif"))
            if save_wbgt:
                expected.append(os.path.join(output_folder, f"WBGT_{key}.tif"))
            if save_ta:
                expected.append(os.path.join(output_folder, f"Ta_{key}.tif"))
            if save_wind:
                expected.append(os.path.join(output_folder, f"Wind_{key}.tif"))
            if all(os.path.exists(p) for p in expected):
                # UTCI/Shadow must also have been written with the requested
                # clip_buildings and tree_transmissivity_pct settings, and UTCI
                # with the requested albedo settings, otherwise the tile is
                # recomputed.
                tagged = [("UTCI", expected[0])]
                if save_shadow:
                    tagged.append(("Shadow", os.path.join(output_folder, f"Shadow_{key}.tif")))
                stale = [os.path.basename(p) for var, p in tagged if clip_state(p, var) != clip_buildings]
                reasons = []
                if stale:
                    reasons.append(f"{', '.join(stale)} not written with clip_buildings={clip_buildings}")
                written_albedo = albedo_state(expected[0])
                wanted_albedo = albedo_tag(albedo_path, albedo_ground, albedo_walls, air_temp_cut)
                if written_albedo != wanted_albedo:
                    reasons.append(f"{os.path.basename(expected[0])} written with albedo settings "
                                   f"{written_albedo}, not {wanted_albedo}")
                stale_trees = [os.path.basename(p) for _, p in tagged
                               if tree_state(p) != tree_tag(tree_transmissivity_pct)]
                if stale_trees:
                    reasons.append(f"{', '.join(stale_trees)} not written with "
                                   f"tree_transmissivity_pct={tree_transmissivity_pct}")
                if not reasons:
                    print(f"[skip] tile {key}: outputs already exist (overwrite=False)")
                    continue
                print(f"[rerun] tile {key}: {'; '.join(reasons)}; recomputing")

        os.makedirs(output_folder, exist_ok=True)

        met_file_data = np.loadtxt(met_file_path, skiprows=1, delimiter=' ', ndmin=2)

        if air_temp_cut and not cut_hours_checked:
            cut_hours_checked = True
            factors = air_temp_cut_factor(met_file_data[:, 2] + met_file_data[:, 3] / 60.0)
            uncut = [f"{int(h):02d}:{int(m):02d}"
                     for h, m, f in zip(met_file_data[:, 2], met_file_data[:, 3], factors) if f == 0.0]
            if uncut:
                print(f"[WARNING] The cool-roof air-temperature cut only applies between 12:00 and "
                      f"18:00 local time; no cut at {', '.join(uncut[:6])}{' ...' if len(uncut) > 6 else ''}")

        compute_utci(
            building_dsm_path,
            tree_path,
            dem_path,
            walls_path,
            aspect_path,
            landcover_path,
            windcoeff_path,
            met_file_data,
            output_folder,
            key,
            selected_date_str,
            save_tmrt=save_tmrt,
            save_svf=save_svf,
            save_kup=save_kup,
            save_kdown=save_kdown,
            save_lup=save_lup,
            save_ldown=save_ldown,
            save_shadow=save_shadow,
            save_wbgt=save_wbgt,
            save_ta=save_ta,
            save_wind=save_wind,
            clip_buildings=clip_buildings,
            albedo_path=albedo_path,
            albedo_walls=albedo_walls,
            albedo_ground=albedo_ground,
            air_temp_cut=air_temp_cut,
            tree_transmissivity_pct=tree_transmissivity_pct,
        )


def thermal_comfort(
    base_path,
    selected_date_str,
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    landcover_filename: Optional[str] = None,
    windcoeff_filename: Optional[str] = None,
    tile_size=3600,
    overlap=20,
    use_own_met=True,
    start_time=None,
    end_time=None,
    data_source_type=None,
    data_folder=None,
    own_met_file=None,
    use_uhi=True,
    hours=None,
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False,
    save_wbgt=False,
    save_ta=False,
    save_wind=False,
    clip_buildings=True,
    albedo_filename: Optional[str] = None,
    baseline_albedo_filename: Optional[str] = None,
    albedo_walls=0.2,
    albedo_ground=0.15,
    tree_transmissivity_pct: Optional[float] = None,
):
    """
    Main function to compute urban thermal comfort using the SOLWEIG-GPU model.

    Args:
        base_path (str): Base directory for outputs and relative raster paths.
        selected_date_str (str): Simulation date in format 'YYYY-MM-DD'
        building_dsm_filename (str): Building+terrain DSM path or filename.
        dem_filename (str): DEM path or filename.
        trees_filename (str): Vegetation DSM path or filename.
        landcover_filename (str, optional): Land cover raster path or filename.
        windcoeff_filename (str, optional): Wind coefficient raster path or filename.
        tile_size (int): Tile size in pixels.
        overlap (int): Overlap between tiles in pixels.
        use_own_met (bool): Use custom meteorological file.
        start_time (str, optional): Start datetime 'YYYY-MM-DD HH:MM:SS'
        end_time (str, optional): End datetime 'YYYY-MM-DD HH:MM:SS'
        data_source_type (str, optional): 'ERA5' or 'wrfout'
        data_folder (str, optional): Folder containing ERA5/WRF NetCDF files
        own_met_file (str, optional): Path to custom meteorological text file
        use_uhi (bool): If True, use UHI-aware ERA5 processing and propagate
            UHI_CYCLE into metfiles. If False, disable it and write uhii = 0.0.
        hours (list[int], optional): UTC hours-of-day to simulate (e.g. [4, 9, 12]).
            If None, every timestep in [start_time, end_time] is used. The window
            must span the requested hours. Ignored when use_own_met is True.
            Each hour gets the same sun position as in a full-day run.
        save_tmrt (bool): Save mean radiant temperature output.
        save_svf (bool): Save sky view factor output.
        save_kup (bool): Save upward shortwave radiation.
        save_kdown (bool): Save downward shortwave radiation.
        save_lup (bool): Save upward longwave radiation.
        save_ldown (bool): Save downward longwave radiation.
        save_shadow (bool): Save shadow maps.
        save_ta (bool): Save diagnostic Ta field.
        save_wind (bool): Save diagnostic wind field.
        clip_buildings (bool): Set building footprints (Building DSM - DEM >= 2 m)
            to NaN in the UTCI and Shadow outputs (default). False keeps the
            rooftop values.
        albedo_filename (str, optional): Ground albedo raster (fractions 0-1) path
            or filename. Replaces the land-cover or default ground albedo.
        baseline_albedo_filename (str, optional): Baseline albedo raster for
            cool-roof scenarios (requires albedo_filename); lowers the forcing air
            temperature by 6 x (mean albedo increase) °C at 12:00 local time,
            scaled down to 18:00 and not applied outside 12:00-18:00.
        albedo_walls (float): Wall albedo (0-1).
        albedo_ground (float): Ground albedo (0-1) used where there is neither
            land cover nor an albedo raster.
        tree_transmissivity_pct (float, optional): Percent (0-100) of the light
            that gets through tree canopies on every date, e.g. 3 for trees in
            leaf all year (tropical cities). None uses a temperate deciduous
            season (leafless outside days 98-299).

    Returns:
        None
    """
    preprocess_dir = preprocess(
        base_path=base_path,
        selected_date_str=selected_date_str,
        building_dsm_filename=building_dsm_filename,
        dem_filename=dem_filename,
        trees_filename=trees_filename,
        landcover_filename=landcover_filename,
        windcoeff_filename=windcoeff_filename,
        tile_size=tile_size,
        overlap=overlap,
        use_own_met=use_own_met,
        start_time=start_time,
        end_time=end_time,
        data_source_type=data_source_type,
        data_folder=data_folder,
        own_met_file=own_met_file,
        use_uhi=use_uhi,
        hours=hours,
        albedo_filename=albedo_filename,
        baseline_albedo_filename=baseline_albedo_filename,
    )

    run_walls_aspect(preprocess_dir)

    calculate_svf(preprocess_dir, patch_option=2, overwrite=False,)

    run_utci_tiles(
        base_path=base_path,
        preprocess_dir=preprocess_dir,
        selected_date_str=selected_date_str,
        tile_keys=None,
        save_tmrt=save_tmrt,
        save_svf=save_svf,
        save_kup=save_kup,
        save_kdown=save_kdown,
        save_lup=save_lup,
        save_ldown=save_ldown,
        save_shadow=save_shadow,
        save_wbgt=save_wbgt,
        save_ta=save_ta,
        save_wind=save_wind,
        clip_buildings=clip_buildings,
        albedo_walls=albedo_walls,
        albedo_ground=albedo_ground,
        tree_transmissivity_pct=tree_transmissivity_pct,
    )
