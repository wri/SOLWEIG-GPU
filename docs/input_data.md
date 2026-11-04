# Input Data

SOLWEIG-GPU requires two main types of input data: geospatial rasters and meteorological forcing data. This guide provides detailed information on how to prepare and format your input data.

!!! tip "No input data for your study area? (New in Version 2)"
    The `build_inputs()` function can download and build the required rasters (Building DSM, DEM, Trees, Landuse) and meteorological data for any location from near-globally available urban datasets, using Google Earth Engine. See the [Quick Start Guide](quickstart.md#example-0-optional-download-input-data-automatically) and [API Reference](api_reference.md#build_inputs).

!!! tip "Outputs in a different folder."
    If you want outputs written to a different directory, set `base_path` to that folder and pass **complete paths** for the Building DSM, DEM, Trees, and land cover (optional) rasters. See [Configuration](configuration.md#base_path).

## Geospatial Rasters

All input rasters must be in **GeoTIFF format** and share the same properties to ensure proper alignment and processing.

### Requirements

!!! Critical Requirements:
    All input rasters must have the following 
    
    - Identical Coordinate Reference System (CRS)
    - Same dimensions (width and height in pixels)
    - Same pixel size (spatial resolution)
    - Proper alignment (same geotransform)

The package includes validation checks that will raise errors if these requirements are not met.

### Required Rasters

#### Building DSM

**Filename:** `Building_DSM.tif` (default)

A Digital Surface Model that includes the height of both buildings and terrain.

**Units:** Meters above a reference datum  


!!! Tip "Creating Building DSM":
    Building DSM can be created by combining:
    
    - LiDAR point cloud data
    - Photogrammetric 3D models
    - Building footprints with height attributes merged with terrain elevation

#### Digital Elevation Model (DEM)

**Filename:** `DEM.tif` (default)

A Digital Elevation Model representing the bare-earth elevation.

**Units:** Meters above a reference datum  

!!! Tip "DEM Sources":
    High-quality DEMs can be obtained from:
    
    - National elevation datasets (e.g., USGS 3DEP for USA)
    - LiDAR-derived bare-earth models
    - SRTM or ASTER GDEM for global coverage (lower resolution)

#### Tree DSM

**Filename:** `Trees.tif` (default)

A Digital Surface Model representing only the vegetation heights (The Tree DSM should represent the height of vegetation canopy above the ground surface, not above the reference datum.). Pixels with no tree coverage should be zero.

Trees are treated as deciduous by default: in leaf from day of year 98 to 299 (3% of the light gets through the canopy) and leafless outside it (50% of the direct sun). Pass `tree_transmissivity_pct` to set the percent of the light that gets through the canopy on every date; `3` keeps evergreen or tropical trees in leaf all year.

**Units:** Meters  

#### Land Cover (Optional)

**Filename:** `Landcover.tif` (optional)

A raster representing land cover types, on exactly the Building DSM grid (same size, pixel size and CRS). If not provided, every ground pixel gets one default surface (albedo `albedo_ground`, emissivity 0.95). The package accepts UMEP-style land cover (see [Urban Land Cover Reclassifier](https://umep-docs.readthedocs.io/en/latest/pre-processor/Urban%20Land%20Cover%20Land%20Cover%20Reclassifier.html)).

Land cover sets each ground pixel's albedo, emissivity and sunlit surface temperature curve, using the classes in `landcoverclasses_2016a.txt`:

| Code | Description |
|------|-------------|
| 1 | Paved (roads, parking, other built-up) |
| 2 | Buildings |
| 5 | Grass |
| 6 | Bare soil |
| 7 | Water |

Codes 3 and 4 (evergreen and deciduous trees in UMEP) are treated as grass, since trees come from the Tree DSM; any other value is treated as bare soil. `preprocess` stops with an error if most pixels are outside 1–7.

**WRI OpenUrban land cover** (codes 110–622) must be converted first, with the same mapping CTCM uses (CIF `open_urban.reclass_map`):

```bash
python -m solweig_gpu.landcover OpenUrban.tif Landcover.tif
```

It writes a Byte GeoTIFF on the same grid: roads, parking and other built-up → 1, buildings → 2, green and public open space → 5, barren → 6, water → 7.

#### Albedo (Optional)

**Filename:** any, e.g. `Albedo.tif` (`albedo_filename`; CLI `--albedo`)

A raster of ground albedo as **fractions from 0 to 1**, e.g. the Cities Indicators Framework `AlbedoCloudMasked` layer (Sentinel-2). On every pixel, roofs included, it replaces the albedo that would otherwise come from the land cover or from `albedo_ground` (default 0.15), as in WRI's CTCM workflow. Land cover, if given, still sets surface temperatures and emissivity.

- **Grid:** any resolution or CRS. If it is not on the Building DSM grid, preprocessing resamples it bilinearly onto that grid (`processed_inputs/albedo_aligned.vrt`) before tiling.
- **Missing data:** NaN or nodata pixels keep the land-cover or default albedo.
- **Units:** values above 1 stop the run with an error; divide a raster stored in percent (0–100) by 100 first.

Albedo only changes reflected shortwave radiation. The ground albedo sets how much sunlight the ground reflects onto a person (upwelling shortwave, Kup); walls reflect with `albedo_walls` (default 0.2). Surface temperatures do not depend on albedo, so a brighter surface raises the daytime Tmrt and UTCI above it. The ground seen from a street pixel stops at the first building, so roof albedo only affects rooftop pixels.

**Baseline albedo for cool-roof scenarios** (`baseline_albedo_filename`; CLI `--baseline_albedo`; requires `albedo_filename`): the mean albedo increase Δα of the albedo raster over the baseline, across the whole domain, sets an air-temperature cut following Krayenhoff et al. (2021): 6 × Δα °C at 12:00 local time (0.1 more albedo gives 0.6 °C cooler air), interpolated to 0.935 × that at 15:00 and 0.647 × at 18:00, with no cut before 12:00 or after 18:00. The cut is subtracted from the forcing air temperature, so it changes both the radiation calculation and UTCI. Preprocessing records it in `processed_inputs/Albedo/air_temp_cut.json` and `run_utci_tiles` applies it. In this model the cut is the only way cool roofs change street-level UTCI.

## Meteorological Forcing Data

SOLWEIG-GPU supports three different types of meteorological data sources. You must choose one of these options for your simulation.

### Option 1: Custom Text File (UMEP Format)

A text file with a specific format generated by the UMEP (Urban Multi-scale Environmental Predictor) tool: <https://umep-docs.readthedocs.io/en/latest/pre-processor/Meteorological%20Data%20MetPreprocessor.html>.

!!! note "Wind direction for directional wind coefficients"
    If you use directional wind-extension coefficients (`ERA_5_z0_find=True` or a `windcoeff_folder`), the meteorological file must contain a wind direction (`Wd`) value for each time step (meteorological wind-from direction: 0=N, 90=E, 180=S, 270=W).

### Option 2: ERA5 Reanalysis Data

NetCDF files from the Copernicus Climate Data Store containing ERA5 reanalysis data: <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview>.

**Required Files:** Both instantaneous and accumulated data files

**Required Variables:**

| Variable | Description | Units |
|----------|-------------|-------|
| t2m | 2-meter air temperature | K |
| d2m | 2-meter dew point temperature | K |
| sp | Surface pressure | Pa |
| u10 | 10-meter U wind component | m/s |
| v10 | 10-meter V wind component | m/s |
| ssrd | Surface solar radiation downwards (accumulated) | J/m² |
| strd | Surface thermal radiation downwards (accumulated) | J/m² |
| fsr | Forecast surface roughness (required only if `ERA_5_z0_find=True`, for directional wind coefficients) | m |

**File Naming:** ERA5 files should be named according to the standard ERA5 naming convention (do not rename the downloaded ERA-5 instantaneous and accumulated data).

#### Downloading ERA-5 for the latest date and your study area

You can fetch the two required files automatically with the bundled
`solweig_gpu.download_era5` helper. It reads your **Building DSM/DEM extent** to
set the download area (so it always matches the city you are modelling) and, by
default, picks the **most recent date** ERA-5 has available.

1. Create a free [CDS account](https://cds.climate.copernicus.eu) and, on the
   [ERA5 single-levels dataset page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download),
   accept the licence at the bottom of the *Download* tab (otherwise requests are rejected).
2. Copy your API token from your [CDS profile](https://cds.climate.copernicus.eu/profile)
   into a `~/.cdsapirc` file:

        url: https://cds.climate.copernicus.eu/api
        key: <YOUR-PERSONAL-ACCESS-TOKEN>

3. Install the client: `pip install cdsapi`
4. Download (uses the latest available date by default):

        python -m solweig_gpu.download_era5 --dsm /path/to/Building_DSM.tif --out data/ERA-5_Forcing_data

   For a specific day, add `--date 2024-08-13`. To fetch only certain hours (UTC),
   add e.g. `--time 6 12 18` (default is all 24 hours). Widen the area with `--margin 1.0`.

This writes `data_stream-oper_stepType-instant.nc` and
`data_stream-oper_stepType-accum.nc` into the output folder — pass that folder as
`data_folder` (with `data_source_type='ERA5'`) to your run.

!!! tip "Higher resolution with ERA5-Land"
    Add `--source era5-land` to download from
    [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land)
    (0.1° / ~9 km) instead of single-levels (0.25° / ~31 km). Accept the
    `reanalysis-era5-land` licence on its dataset page first. The output files,
    their names, and the rest of the workflow are identical — `data_source_type`
    stays `'ERA5'`.

    ERA5-Land accumulates `ssrd`/`strd` from 00 UTC and resets daily, so the
    helper de-accumulates them into per-hour totals (matching single-levels)
    before writing. Because that needs a contiguous day, `era5-land` always
    downloads all 24 hours (any `--time` subset is ignored).

!!! warning "ERA-5 is not real time"
    The ERA5T preliminary stream lags real time by about **5 days**, so the latest
    available date is roughly *today − 5 days* (requesting today or yesterday will
    fail). The helper picks a safe recent date automatically and steps back a day
    if the most recent one is not yet published.

### Option 3: WRF Output Files

Output files from the Weather Research and Forecasting (WRF) model.

**File Format:** NetCDF (wrfout files)

**Filename Pattern:** Must follow one of these patterns:

- `wrfout_d0X_YYYY-MM-DD_HH_MM_SS` (recommended, works across all operating systems)
- `wrfout_d0X_YYYY-MM-DD_HH:MM:SS`
- `wrfout_d0X_YYYY-MM-DD_HH`

Where `X` is the domain number (1-9).

**Required Variables:** The package automatically extracts the required meteorological variables from the WRF output files.

!!! warning "Time Specification."
    When using WRF data, you must specify `start_time` and `end_time` in **UTC**. The package will automatically convert to local time.

## Data Preparation Tips

### Coordinate Reference System

Choose an appropriate projected coordinate system for your study area. UTM zones are commonly used for urban-scale studies.

### Temporal Resolution

Currently, SOLWEIG-GPU is tested and optimized for **hourly data**. Sub-hourly or multi-hourly data may require modifications to the code.

## Sample Data

A complete sample dataset is available for download to help you understand the expected data format and structure.

**Download:** [Sample Data on Zenodo](https://doi.org/10.5281/zenodo.21081622)

The sample dataset includes:

- Example input rasters for Austin, Texas
- Meteorological forcing data in all three formats (custom text, ERA5, WRF) 

!!! Tip:
    We recommend starting with the sample data to familiarize yourself with the package before processing your own data. For a quick test, download only ERA5 or ownmet data, since the entire forcing data set is too large (which includes high-resolution WRF data). 
