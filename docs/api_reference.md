># API Reference

This section provides a detailed reference for the functions and classes in the SOLWEIG-GPU package.

**Entry points:**

- **`thermal_comfort(...)`** – One-shot: runs preprocessing, wall/aspect, and UTCI in one call. Use this for normal runs (CLI and GUI call this).
- **`preprocess(...)`**, **`run_walls_aspect(preprocess_dir)`**, **`run_utci_tiles(...)`** – Staged execution: call these when you need to run preprocessing once, then wall/aspect, then UTCI (optionally for a subset of tiles). See [Developer Guide – Pipeline stages](developer_guide.md#pipeline-stages) and the repository’s [REFACTORING.md](../REFACTORING.md).

---

## `thermal_comfort()`

Main function to run a full SOLWEIG-GPU simulation (preprocess → wall/aspect → UTCI). Same behavior as before the staged API; CLI and GUI are unchanged.

```python
from solweig_gpu import thermal_comfort

thermal_comfort(
    base_path,
    selected_date_str,
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    landcover_filename: Optional[str] = None, 
    tile_size=3600, 
    overlap = 20,
    use_own_met=True,
    start_time=None, 
    end_time=None, 
    data_source_type=None, 
    data_folder=None,
    own_met_file=None,
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False,
    clip_buildings=True,
    albedo_filename=None,
    baseline_albedo_filename=None,
    albedo_walls=0.2,
    albedo_ground=0.15,
    tree_transmissivity_pct=None
)
```

### Parameters

-   `base_path` (str): Base directory for `output_folder/` and `processed_inputs/`; also used to resolve relative raster paths. To write outputs elsewhere, set this to that directory and pass complete paths for the raster parameters.
-   `selected_date_str` (str): The date for the simulation in `YYYY-MM-DD` format.
-   `building_dsm_filename` (str): Path or filename of the Building DSM raster (relative to `base_path`, or a complete path). Defaults to `'Building_DSM.tif'`.
-   `dem_filename` (str): Path or filename of the DEM raster. Defaults to `'DEM.tif'`.
-   `trees_filename` (str): Path or filename of the Tree DSM raster. Defaults to `'Trees.tif'`.
-   `landcover_filename` (str, optional): Path or filename of the land cover raster, in SOLWEIG codes (1 paved, 2 buildings, 5 grass, 6 bare soil, 7 water) and on the Building DSM grid. Convert OpenUrban land cover first with `python -m solweig_gpu.landcover`. Defaults to `None`. See [Input Data – Land Cover](input_data.md#land-cover-optional).
-   `tile_size` (int): The size of the tiles in pixels. Defaults to `3600`.
-   `overlap` (int): The overlap between tiles in pixels. Defaults to `20`.
-   `use_own_met` (bool): Whether to use a custom meteorological file. Defaults to `True`.
-   `start_time` (str, optional): The start time of the simulation in `YYYY-MM-DD HH:MM:SS` format (UTC). Required if `use_own_met` is `False`.
-   `end_time` (str, optional): The end time of the simulation in `YYYY-MM-DD HH:MM:SS` format (UTC). Required if `use_own_met` is `False`.
-   `data_source_type` (str, optional): The type of meteorological data source (`'ERA5'` or `'wrfout'`). Required if `use_own_met` is `False`.
-   `data_folder` (str, optional): The directory containing the meteorological data files. Required if `use_own_met` is `False`.
-   `own_met_file` (str, optional): The path to the custom meteorological file. Required if `use_own_met` is `True`.
-   `save_tmrt` (bool, optional): Whether to save the Mean Radiant Temperature output. Defaults to `True`.
-   `save_svf` (bool, optional): Whether to save the Sky View Factor output. Defaults to `False`.
-   `save_kup` (bool, optional): Whether to save the upwelling shortwave radiation output. Defaults to `False`.
-   `save_kdown` (bool, optional): Whether to save the downwelling shortwave radiation output. Defaults to `False`.
-   `save_lup` (bool, optional): Whether to save the upwelling longwave radiation output. Defaults to `False`.
-   `save_ldown` (bool, optional): Whether to save the downwelling longwave radiation output. Defaults to `False`.
-   `save_shadow` (bool, optional): Whether to save the shadow map output. Defaults to `False`.
-   `clip_buildings` (bool, optional): Whether to set building footprints (Building DSM − DEM ≥ 2 m) to NaN in the UTCI and Shadow outputs. `False` keeps the rooftop values. Defaults to `True`. See [Outputs – Building clipping](outputs.md#building-clipping).
-   `albedo_filename` (str, optional): Path or filename of a ground albedo raster (fractions 0–1, any grid). It replaces the land-cover or default ground albedo. Defaults to `None`. See [Input Data – Albedo](input_data.md#albedo-optional).
-   `baseline_albedo_filename` (str, optional): Path or filename of the baseline albedo raster for cool-roof scenarios; requires `albedo_filename`. The forcing air temperature is lowered by 6 × (mean albedo increase) °C at 12:00 local time, less until 18:00, and not at all outside 12:00–18:00. Defaults to `None`.
-   `albedo_walls` (float, optional): Wall albedo (0–1). Defaults to `0.2`.
-   `albedo_ground` (float, optional): Ground albedo (0–1) used where there is neither land cover nor an albedo raster. Defaults to `0.15`.
-   `tree_transmissivity_pct` (float, optional): Percent (0–100) of the light, direct sun and diffuse sky light, that gets through tree canopies on every date, e.g. `3` for trees in leaf all year as in tropical cities. `None` (default) uses a temperate deciduous season: 3% on day of year 98–299, and outside it 50% of the direct sun (3% of the diffuse light). See [Outputs – Tree Transmissivity Tag](outputs.md#tree-transmissivity-tag).

---

## `preprocess()`

Runs only preprocessing: validates rasters, creates tiles, and prepares metfiles. Use when you want to call the pipeline in stages.

```python
from solweig_gpu import preprocess

preprocess_dir = preprocess(
    base_path="/path/to/data",
    selected_date_str="2020-08-13",
    building_dsm_filename="Building_DSM.tif",
    dem_filename="DEM.tif",
    trees_filename="Trees.tif",
    landcover_filename=None,
    tile_size=1000,
    overlap=100,
    use_own_met=True,
    own_met_file="/path/to/met.txt",
    preprocess_dir=None,  # optional; default is base_path/processed_inputs
    albedo_filename=None,  # optional ground albedo raster (0-1)
    baseline_albedo_filename=None,  # optional; enables the cool-roof air-temperature cut
)
```

**Returns:** Path to the preprocessing directory (string).

**Parameters:** Same as the first part of `thermal_comfort` (base_path, selected_date_str, raster filenames including `albedo_filename` and `baseline_albedo_filename`, tile_size, overlap, met options). Optional `preprocess_dir` overrides the default output directory. The albedo raster is resampled onto the Building DSM grid if needed and tiled into `{preprocess_dir}/Albedo/`; with a baseline, the air-temperature cut is recorded in `{preprocess_dir}/Albedo/air_temp_cut.json`. Without `albedo_filename`, an `Albedo/` folder left by an earlier run is removed.

---

## `run_walls_aspect()`

Computes wall heights and aspects for all tiles in the preprocessing directory. Call after `preprocess()`.

```python
from solweig_gpu import run_walls_aspect

run_walls_aspect(preprocess_dir)
```

**Parameters:**

-   `preprocess_dir` (str): Path returned by `preprocess()` (contains Building_DSM/, DEM/, Trees/, etc.).

**Returns:** None. Writes to `{preprocess_dir}/walls/` and `{preprocess_dir}/aspect/`.

---

## `run_utci_tiles()`

Runs the SOLWEIG/UTCI computation for tiles. Call after `preprocess()` and `run_walls_aspect()`.

```python
from solweig_gpu import run_utci_tiles

run_utci_tiles(
    base_path="/path/to/data",
    preprocess_dir=preprocess_dir,
    selected_date_str="2020-08-13",
    tile_keys=None,       # None = all tiles; or ["0_0", "1000_0"] for a subset
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False,
    clip_buildings=True,
    overwrite=False,
    albedo_walls=0.2,
    albedo_ground=0.15,
    air_temp_cut=None,
    tree_transmissivity_pct=None,
)
```

**Parameters:**

-   `base_path` (str): Base directory; `output_folder/` is created under it.
-   `preprocess_dir` (str): Path returned by `preprocess()`.
-   `selected_date_str` (str): Simulation date `YYYY-MM-DD`.
-   `tile_keys` (list, optional): If `None`, process all tiles. If a list (e.g. `["0_0", "1000_0"]`), process only those tile keys.
-   `save_tmrt`, `save_svf`, `save_kup`, `save_kdown`, `save_lup`, `save_ldown`, `save_shadow`, `clip_buildings`, `albedo_walls`, `albedo_ground`, `tree_transmissivity_pct`: Same as in `thermal_comfort()`. The albedo raster, if any, is picked up from `{preprocess_dir}/Albedo/`.
-   `overwrite` (bool, optional): If `False` (default), skip tiles whose requested outputs already exist, so an interrupted run resumes. A tile whose UTCI or Shadow was written with a different `clip_buildings` or `tree_transmissivity_pct` setting (or before the latter was recorded), or whose UTCI was written with different albedo settings, is recomputed. `True` recomputes every tile.
-   `air_temp_cut` (float, optional): Cool-roof air-temperature cut at 12:00 local time, in °C. `None` (default) uses the cut `preprocess()` derived from `baseline_albedo_filename`, or no cut if there was none.

**Returns:** None. Writes GeoTIFFs to `{base_path}/output_folder/{tile_key}/`.

