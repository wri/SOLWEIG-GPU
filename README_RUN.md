# Running a city with `run.py`

`run.py` runs the whole SOLWEIG-GPU pipeline for one city and one date:
1. It downloads the ERA5 weather and cuts the rasters into tiles.
2. It builds the geometry caches.
3. It computes UTCI, plus any other outputs you ask for, tile by tile.
4. It stitches the tiles into city-wide rasters.

Every path and setting is in **`config.toml`**, so you shouldn't need to edit the Python files.

| File | What it is |
|---|---|
| `config.toml` | All paths and settings of a run. Copy it to set up another city or date. |
| `run.py` | The entry point: reads a config file and runs the pipeline. |
| `processing.py` | The pipeline itself, and the checks on the config file. |

## Quick start

```bash
# 1. Once per machine: get access to the ERA5 data (see "ERA5 access" below).
# 2. Edit config.toml: base_path, date, hours, ...
# 3. Run it, from any folder:
.venv/bin/python run.py                  # uses config.toml next to run.py
.venv/bin/python run.py bhopal.toml      # or another config file
```

A long run can keep going after you log out:

```bash
nohup .venv/bin/python run.py > run.log 2>&1 &
tail -f run.log
```

## What a run does

| Step | Function | Writes to (inside `base_path`) | On a rerun |
|---|---|---|---|
| 0. Weather | `download_era5` | `<era5_folder>/` | Skipped if the folder already covers the time window |
| Land cover | `openurban_to_solweig` | `<landcover>` | Redone |
| 1. Tiles and forcing | `preprocess` | `processed_inputs/` | Redone |
| 2. Walls and aspect | `run_walls_aspect` | `processed_inputs/walls/`, `aspect/` | Existing tiles are reused |
| 2. Sky view factor | `calculate_svf` | `processed_inputs/SVF/` | Existing tiles are reused |
| 3. UTCI | `run_utci_tiles` | `output_folder/<tile>/` | Finished tiles are skipped |
| 4. Mosaic | `mosaic_tiles` | `output_folder/mosaic/` | Redone |

A city's first run also builds the geometry caches (step 2) for every tile; later runs reuse them.

For scale, Mumbai has 231 tiles of 2000 px:
- Step 3 took about 25 s per tile on an RTX 4060 Ti.
- Its tiles and mosaic take roughly 30 GB.

## Before the first run

### Python

Use the project's virtual environment, `.venv/bin/python`. Another environment needs:
- `solweig_gpu`;
- `cdsapi`;
- Python 3.11 or newer. On Python 3.10, also run `pip install tomli`.

The model runs on an NVIDIA GPU when PyTorch finds one, otherwise on the CPU, which is much slower.

### Input rasters

Put the rasters in one folder; that folder is `base_path` in the config.

| Setting | In `config.toml` | Required | Content |
|---|---|---|---|
| `building_dsm` | `Building_DSM.tif` | yes | Height of buildings and terrain (m). Its extent also sets the ERA5 download area. |
| `dem` | `DEM.tif` | yes | Bare-ground elevation (m). |
| `trees` | `Trees.tif` | yes | Canopy height above the ground (m), 0 where there are no trees. |
| `openurban` | `OpenUrban.tif` | no | WRI OpenUrban land cover. It is converted to SOLWEIG codes into `landcover` on every run. |
| `landcover` | `Landcover.tif` | no | Land cover in SOLWEIG codes: 1 paved, 2 buildings, 5 grass, 6 bare soil, 7 water. |
| `albedo` | `Albedo.tif` | no | Ground albedo as fractions 0–1, not percent. Any grid works; it is resampled onto the DSM grid. |

The DSM, DEM, trees and land cover must share one grid: the same CRS, size and pixel size. [docs/input_data.md](docs/input_data.md) has the details.

### ERA5 access

The weather comes from the Copernicus Climate Data Store (CDS). Set this up once per machine:

1. Create a free account at <https://cds.climate.copernicus.eu>.
2. Accept the licence for each dataset you use; requests are rejected otherwise. It is at the bottom of the *Download* tab of [ERA5 single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download), and of [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download) if you use it.
3. Copy the API token from your [CDS profile](https://cds.climate.copernicus.eu/profile) into `~/.cdsapirc`:

   ```
   url: https://cds.climate.copernicus.eu/api
   key: <YOUR-PERSONAL-ACCESS-TOKEN>
   ```

A download waits in the CDS queue, sometimes for several minutes.

If the ERA5 folder already holds the data, for example copied from another machine, the run never contacts the CDS.

## The config file

`config.toml` is a [TOML](https://toml.io) file. Text goes in quotes, `true`/`false` are lower case, lists go in brackets, and `#` starts a comment.

**Required and optional settings.** Every setting is required except those marked *optional*. To turn an optional setting off, leave it out or comment it out.

**Checking.** The run checks the whole file before it starts and lists every unknown, missing or mistyped setting. A typo therefore stops the run instead of being ignored.

**Paths.** Paths in `[paths]` are relative to `base_path`; absolute paths work too. A relative `base_path` is relative to the config file's folder.

### `[paths]`

| Setting | Example | Meaning |
|---|---|---|
| `base_path` | `"data/mumbai/"` | Folder with the input rasters. `processed_inputs/` and `output_folder/` are created in it. |
| `building_dsm`, `dem`, `trees` | `"Building_DSM.tif"` | The required input rasters (see above). |
| `openurban` | `"OpenUrban.tif"` | Optional. OpenUrban land cover to convert into `landcover`. Leave it out if `landcover` already holds SOLWEIG codes. |
| `landcover` | `"Landcover.tif"` | Optional. Land cover in SOLWEIG codes. |
| `albedo` | `"Albedo.tif"` | Optional. Ground albedo raster, fractions 0–1. |
| `era5_folder` | `"era-5-single-levels"` | Where the ERA5 files are downloaded. It holds one day at a time. |

### `[time]`

| Setting | Example | Meaning |
|---|---|---|
| `date` | `"2025-05-29"` | Simulation date. The download fetches this UTC day. |
| `start`, `end` | `"2025-05-29 00:00:00"`, `"2025-05-29 23:00:00"` | Time window in UTC. Keep it inside the UTC day `date`, and make it span every hour in `hours`. |
| `hours` | `[4, 9, 12]` | Optional. The UTC hours of the day to simulate, one output band each. Leave it out to simulate every hour of the window. |

- **Time zones.** `hours`, `start` and `end` are in **UTC**, but output bands and mosaic file names use local time. For India (IST = UTC+5:30), 4, 9 and 12 UTC are 09:30, 14:30 and 17:30 IST.
- **Recent dates.** ERA5 is published about 5 days behind real time, so a more recent date fails to download.
- **One day of data.** The download covers one UTC day, which is enough for daytime hours. A full local day, or night-time hours, would also need the neighbouring days, and `run.py` does not fetch those.

### `[era5]`

| Setting | Example | Meaning |
|---|---|---|
| `source` | `"era5"` | `"era5"`: ERA5 single levels, 0.25° (~31 km), with data over land and sea. `"era5-land"`: ERA5-Land, 0.1° (~9 km), but with **no data over the sea**. |

With ERA5-Land, coastal tiles get -999 forcing and -999 UTCI, so use `"era5"` for coastal cities.

### `[tiling]`

| Setting | Example | Meaning |
|---|---|---|
| `tile_size` | `2000` | Tile size in pixels. |
| `overlap` | `20` | Overlap between tiles, in pixels. |
| `overwrite_geometry` | `false` | `true` rebuilds the walls/aspect and SVF caches. Do this after changing the input rasters, `tile_size` or `overlap`; the caches only match the tiles they were built for. |

### `[model]`

| Setting | Example | Meaning |
|---|---|---|
| `tree_transmissivity_pct` | `3` | Optional. Percent (0–100) of the light, direct and diffuse, that gets through tree canopies on every date. `3` is a dense canopy in leaf, as in Indian cities. |

If you leave `tree_transmissivity_pct` out, trees follow a temperate season:
- from 7 April to 26 October they are in leaf, and 3% of the light gets through;
- the rest of the year they are leafless, and 50% of the direct sun gets through.

### `[outputs]`

| Setting | Example | Meaning |
|---|---|---|
| `save_tmrt`, `save_svf`, `save_kup`, `save_kdown`, `save_lup`, `save_ldown`, `save_shadow` | `false` (`save_shadow = true`) | Extra outputs per tile: mean radiant temperature, sky view factor, upward/downward shortwave, upward/downward longwave, and shadow. UTCI is always saved. |
| `clip_buildings` | `false` | `true`: building footprints are NaN in UTCI and Shadow. `false`: rooftop values are kept. |
| `overwrite` | `false` | `false`: finished tiles are skipped, so a run can resume. `true`: every tile is recomputed. |
| `split_bands` | `true` | Mosaic layout. `true`: one file per variable and hour, e.g. `UTCI_2025-05-29_0930.tif`. `false`: one multi-band file per variable, e.g. `UTCI.tif`. |

## Outputs

```
<base_path>/
├── era-5-single-levels/       ERA5 forcing: two NetCDF files for the date
├── processed_inputs/          tiles, forcing and caches
│   ├── Building_DSM/ DEM/ Trees/ Landcover/ Albedo/   raster tiles
│   ├── walls/ aspect/ SVF/    geometry caches, reused by later runs
│   └── metfiles/              forcing per tile and date
└── output_folder/
    ├── 0_0/ 0_2000/ ...       one folder per tile: UTCI_0_0.tif, Shadow_0_0.tif, ...
    └── mosaic/                city-wide rasters, e.g. UTCI_2025-05-29_0930.tif
```

- **Tile names** are pixel offsets (`<x>_<y>`) in the input rasters.
- **Bands:** a tile output has one band per simulated hour. Each band's `Time` metadata item holds the local time.
- **Shadow values:** 1 in full sun and 0 in building shade. Under trees it is the share of sun that gets through the canopy, e.g. 0.03.
- **Recorded settings:** UTCI and Shadow record the `clip_buildings`, albedo and `tree_transmissivity_pct` settings they were computed with, in metadata items. See [docs/outputs.md](docs/outputs.md).

## Rerunning

- **After an interrupted run**, run it again with the same config; finished tiles are skipped. If the run was killed while writing a tile, delete that tile's folder in `output_folder/` first, because its files may be incomplete.
- **After changing `clip_buildings`, albedo or `tree_transmissivity_pct`**, a rerun recomputes the affected tiles automatically.
- **After changing anything else that affects the results** (`[time]`, `[era5] source`, the input rasters or the tiling), first move the old results away, e.g. `mv output_folder output_folder_2025-05-29`, or set `overwrite = true`.
  - **Why:** tile outputs don't record these settings, so finished tiles from the previous run would otherwise be kept as they are.
  - **Moving is cleaner:** the mosaic folder also keeps files for hours you no longer simulate.
  - **ERA5:** the folder is refreshed by itself whenever it doesn't cover the new window.
- **After changing the input rasters, `tile_size` or `overlap`**, also set `overwrite_geometry = true` for one run, or delete `processed_inputs/walls`, `aspect` and `SVF`.
- **For another city**, copy `config.toml` (e.g. to `bhopal.toml`), change `base_path` and the other settings, and run `.venv/bin/python run.py bhopal.toml`. Each city keeps its own `processed_inputs/`, `output_folder/` and ERA5 folder inside its `base_path`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Problems in the config file: ...` | The listed settings are unknown (a typo), missing, or the wrong type, e.g. `"false"` in quotes. Fix them in the config file. |
| The ERA5 download fails | Check `~/.cdsapirc` and that the dataset's licence is accepted. Dates within about 5 days of today are not published yet; the run does not fall back to an earlier day. |
| `The ERA5 download for ... does not cover ...` | `start`/`end` reach outside the UTC day `date`. Keep the window inside it. |
| Coastal tiles have UTCI = -999 | The forcing came from ERA5-Land, which has no data over the sea. Set `source = "era5"` (the next run downloads it) and recompute the tiles (see *Rerunning*). |
| Hours missing with `hours` left out | The run covers the local day, but the download covers only the UTC day `date`. For India, the local hours before 05:30 fall on the previous UTC day, so they are missing. |
| Results look like an older run | `overwrite = false` kept the tiles of a run with another date, hours or forcing. See *Rerunning*. |

## Further reading

- [docs/input_data.md](docs/input_data.md): input rasters and ERA5 data in detail.
- [docs/outputs.md](docs/outputs.md): output files and their metadata items.
- [docs/api_reference.md](docs/api_reference.md): the library functions that `processing.py` calls.
- `solweig_gpu/download_era5.py`: the ERA5 downloader. It also runs on its own: `python -m solweig_gpu.download_era5 --help`.
