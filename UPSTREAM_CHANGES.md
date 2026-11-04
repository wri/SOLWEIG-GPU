# SOLWEIG-GPU — proposed changes from this fork

This document describes every source change in this tree relative to unmodified
SOLWEIG-GPU (baseline copy taken ~2026-06-18). It is written for the upstream
maintainer: what changed, why, the measured speedups, and the evidence that the
outputs are unchanged.

**Merged with upstream v2.0.0.** On 2026-11-04 this work was merged with upstream
`main` at `1fec79a` (v2.0.0, 62 commits after the `38b451d` baseline). Every
measurement and "unchanged output" claim below was made against `38b451d`. After
the merge, the harness was re-run on synthetic tiles (§7) against both `38b451d`
and v2.0.0: walls, aspect and the shadow matrices are bit-identical, and the `svf*`
rasters differ only by float32 rounding (max 4.1e-6), except that `svfN*` now
differ from `38b451d` by design. The merge changes four things:

- **svfN fix.** Upstream fixed the north test in `svf_calculator`:
  `270 <= azimuth < 90` is always false, so `svfN`, `svfNveg` and `svfNaveg`
  stayed 0. The fix is carried into the rewritten accumulation loop (§1.4), so
  these three rasters now differ from the baseline by design, as they do in
  v2.0.0. The other `svf*` rasters, the shadow matrices and walls/aspect are
  unaffected.
- **Bush-branch slice.** v2.0.0 fixed the off-by-one of §3.6 the same way.
- **GUI removed.** Upstream deleted `solweig_gpu_gui.py`, and with it the GUI
  controls for `clip_buildings` and the albedo inputs. Use the Python API or the
  CLI.
- **Wind coefficients.** Upstream replaced the single wind-coefficient raster with
  12 directional rasters (`windcoeff_folder` on `preprocess`; `ERA_5_z0_find` and
  `build_wind_ext_coeff` on `thermal_comfort`). The stale-folder cleanup of §4
  removes `WindCoeff/` when no wind coefficients are given.

Three themes:

1. **Performance** — the SVF/shadow-casting and walls/aspect stages were rewritten
   around GPU batching, `torch.compile`, and vectorized scipy kernels. End-to-end
   SVF per tile is **7–20× faster** and walls+aspect **~21× faster** on the test
   hardware, with outputs verified unchanged (details below).
2. **Correctness fixes** — date-aware metfile selection, a single-timestep metfile
   crash fix, a small vectorization, the sun position of runs that keep only
   some hours, and the land-cover codes the solver sees on the GPU.
3. **New capability & tooling** — a one-command ERA-5 forcing downloader, geometry
   cache reuse across runs, a switch for the building-footprint mask on the
   UTCI/Shadow outputs, albedo inputs (albedo raster, wall/ground albedo, cool-roof
   air-temperature cut), a tree light-transmissivity setting, OpenUrban land-cover conversion
   with a land-cover code check, and an A/B verification harness that backs every
   "unchanged output" claim in this document.

**Measurement environment** (all numbers in this doc): NVIDIA GeForce RTX 4060 Ti
16 GB, torch 2.12.1+cu130 (CUDA 13.0), Python 3.14.4, Linux; measured 2026-07-02
with `patch_option=2` (153 sky patches).

## At a glance

| Area | Files | Change | Measured effect | Output equivalence |
|---|---|---|---|---|
| SVF / shadow casting | `solweig_gpu/shadow.py` | batched multi-azimuth march, `torch.compile`-fused steps, schedule-driven stepping, march ceiling, precomputed annulus weights, tunable npz compression | 1020² tile: ~34.5 s → **4.66 s** (7.4×); 2020² tile: ~422 s → **20.6 s** (20.5×) | shadow/veg matrices **bit-identical**; the 15 `svf*` rasters differ only in float32 last-bit rounding (max ~4e-6) and are *closer* to the float64 reference |
| Walls & aspect | `solweig_gpu/walls_aspect.py` | `findwalls` → `maximum_filter`; aspect → cached rotated filters + batched `conv2d` (GPU) / `correlate` (CPU); skip-if-exists caching | 16 tiles: 63.6 s → **3.0 s** (~21×) | **bit-identical** GeoTIFFs (data + geo metadata) |
| Metfile handling fixes | `solweig_gpu/utci_process.py`, `solweig_gpu/solweig_gpu.py` | date-aware metfile mapping; `ndmin=2`; `diffsh` vectorization | fixes wrong-date selection and a 1-timestep crash | `diffsh` change bit-identical |
| Shadow march bug fix | `solweig_gpu/shadow.py` | bush-branch slice off-by-one in the per-patch `shadow()` (§3.6) | tiles with `bush.max() > 0` no longer crash | bush-free tiles unaffected (branch never runs); regression tests in `tests/test_shadow_bush.py` |
| Sun time with `hours` | `solweig_gpu/sun_position.py` | the sun is placed half the *native* step (at most 1 h) before each metfile row, not half the gap between the rows that `hours` kept (§3.7) | `hours=[4, 9, 12]` no longer puts the sun 2.5 h early; a kept hour gets the same sun as in a full-day run | contiguous hourly and 30-min metfiles bit-identical; runs with gaps between rows or a single row change (intended) |
| Land-cover codes on the GPU | `solweig_gpu/utci_process.py` | the solver gets the recoded land cover (trees → grass, invalid → bare soil), not the raw codes (§3.8) | GPU runs no longer treat land-cover code 3 as water; they match CPU runs | unchanged without land cover or without code 3 |
| ERA-5 downloader | `solweig_gpu/download_era5.py` (new), `setup.py`, `docs/input_data.md` | auto-bbox from the DSM, latest-available-date default, ERA5 & ERA5-Land | one-command forcing download | n/a (new feature) |
| Building clipping option | `solweig_gpu/utci_process.py`, `solweig_gpu/solweig_gpu.py`, `solweig_gpu/cli.py`, docs | `clip_buildings` (default on) NaNs building footprints in UTCI **and Shadow**; outputs tagged `CLIP_BUILDINGS=YES/NO`; resume recomputes tiles written with the other setting (§4) | UTCI unchanged by default; **Shadow gains NaN roofs by default** (was never masked) | clip on: UTCI bit-identical, Shadow bit-identical off buildings; clip off: Shadow bit-identical, UTCI bit-identical off buildings |
| Albedo inputs | `solweig_gpu/albedo.py` (new), `solweig_gpu/preprocessor.py`, `solweig_gpu/utci_process.py`, `solweig_gpu/solweig_gpu.py`, `solweig_gpu/cli.py`, docs | optional ground albedo raster (resampled to the DSM grid), `albedo_walls` / `albedo_ground`, cool-roof air-temperature cut from a baseline albedo raster; UTCI tagged `ALBEDO=...`; resume recomputes tiles written with other albedo settings (§4) | new scenarios: per-pixel albedo, cool-roof runs | defaults bit-identical; a constant-0.15 albedo raster is bit-identical to no raster (`tests/test_albedo.py`, Mumbai tile) |
| Tree transmissivity option | `solweig_gpu/utci_process.py`, `solweig_gpu/solweig_gpu.py`, docs | `tree_transmissivity_pct` (default `None`: the temperate season) sets the percent of light, direct and diffuse, that gets through tree canopies on every date; UTCI/Shadow tagged `TREE_TRANSMISSIVITY_PCT=...`; resume recomputes tiles written with another value (§4) | correct tree shade for tropical cities outside 7 Apr – 26 Oct; any canopy density | default bit-identical; `3` bit-identical to trees in leaf all year |
| OpenUrban land cover | `solweig_gpu/landcover.py` (new), `solweig_gpu/preprocessor.py`, docs | `python -m solweig_gpu.landcover` converts OpenUrban to SOLWEIG codes (CIF mapping); `preprocess` rejects land cover not in SOLWEIG codes and drops stale `Landcover/` / `WindCoeff/` folders (§4) | OpenUrban no longer runs as bare soil everywhere | SOLWEIG-coded land cover unchanged |
| Verification harness | `tools/ab_harness/` (new, 11 files) | dump / compare / profile current-vs-original, from any checkout; synthetic test tiles | evidence for every claim above | n/a (tooling) |
| Packaging | `requirements.txt`, `requirements-lock.txt` (new) | pins for Python 3.14 (pandas 3.0.4 cp314 segfault) | reproducible env | n/a |

Everything performance-related is **flag-gated with an environment-variable escape
hatch** that restores the original code path (see [Tuning & compatibility
reference](#6-tuning--compatibility-reference)), which is also how each
optimization was A/B-verified in isolation.

---

## 1. SVF / shadow casting (`solweig_gpu/shadow.py`)

`svf_calculator` dominates preprocessing time. The original implementation marches
one sky patch at a time (153 calls to `shadow()` per tile), each march step
launching ~15 small elementwise CUDA kernels, with several host↔device
synchronizations per step. The rewrite keeps the numerical recipe identical and
restructures the execution:

### 1.1 Batched multi-patch march — `shadow_batch()` (new)

All azimuths of one altitude band now march together as a `(P, H, W)` batch
(`SOLWEIG_BATCH_MARCH`, default on; `SOLWEIG_BATCH_SIZE`, default 16). The
original integer slice-and-zero-fill `temp[dst] = src[srcwin] - dz` is replaced by
the equivalent integer gather `torch.where(valid, a[r+dx, c+dy] - dz, 0)`, which
is exact (no interpolation), and a per-lane active mask handles ragged march
termination — needed because `vbshvegsh` accumulates every step, so terminated
lanes must be frozen exactly where the original loop stopped them.

- **Bush-free tiles only**: tiles where `bush.max() > 0` automatically fall back to
  the original per-patch `shadow()` path, which is kept fully functional (its bush
  branch carried an upstream off-by-one that crashed on the first bush-bearing tile;
  fixed, see §3.6).
- **Fast freeze** (`SOLWEIG_FAST_FREEZE`, default on, auto-disabled when
  `a.min() < 0`): with non-negative elevations, a terminated lane's `f/sh/vegsh`
  provably cannot move on later steps, so only the `vbshvegsh` accumulator needs
  the freeze mask — 3 of 4 per-step `(P,H,W)` `torch.where` masks are skipped.
  Bit-identical; the full 4-way freeze remains as the fallback.
- **Schedule-driven stepping** (`SOLWEIG_BATCH_SCHEDULE`, default on): `dx/dy/dz`
  and lane liveness are closed-form in (lane, step), so the whole march schedule is
  precomputed as `(S, P)` tensors using the same float32 ops as the stepwise loop
  (bit-identical per step), and the step count is derived with **one** host
  synchronization per batch instead of a `bool(active.any())` sync per step.
- **VRAM-aware batch clamp** (`SOLWEIG_BATCH_CLAMP`, default on): lane count is
  capped from `torch.cuda.mem_get_info()` headroom (a march step holds ~14 live
  float32 planes per lane), so large tiles cannot OOM; partitioning a batch cannot
  change any lane's output, only throughput.

### 1.2 Kernel fusion via `torch.compile`

- `_shadow_core()` (new): the ~8 memory-bound elementwise ops of one march step,
  extracted so `torch.compile` can fuse them (used by both the per-patch and the
  batched path; `SOLWEIG_COMPILE_MARCH`, default on when CUDA is available).
- `_batch_step_fast()` / `_batch_step_full()` (new): one *entire* batched march
  step — per-lane shift indices, bounds mask, three shifted gathers, core update,
  freeze — compiled as a single graph per `(P,H,W)` shape
  (`SOLWEIG_COMPILE_BATCH`, default on). This collapses ~180 eager kernel launches
  per step into a handful.
- Every op involved is elementwise/gather/compare with no fusable multiply-add
  pattern, so the compiler cannot FMA-contract anything: **compiled results are
  bit-identical to eager** (this was verified explicitly, see §5).
- Compilation is wrapped in `try/except` with `torch._dynamo.config.suppress_errors
  = True`: any compile failure falls back to eager for that shape instead of
  crashing an unattended run. Graphs compile once per process and are replayed for
  every subsequent step/tile; the dynamo cache limit is raised so tail-batch shapes
  never evict hot graphs.

### 1.3 March-length ceiling (`SOLWEIG_MARCH_CEILING`, default on)

The original loop marches until `dz` exceeds `amaxvalue`, an **absolute**
elevation. The shadow test only compares *height differences*, so once `dz`
exceeds the true surface height **range** — `max(amaxvalue, vegdem.max()) −
a.min()` — every remaining step is a provable no-op for the binarized outputs.
Capping the march there is bit-identical while dropping a long tail of wasted
steps (significant when the DEM sits at high absolute elevation, e.g. a city at
500 m ASL would march ~500 m of dead height). Elevation values themselves are
untouched.

### 1.4 Accumulation-loop cleanups in `svf_calculator`

- **Annulus weights precomputed per altitude band**: `annulus_weight(k, ·)` is a
  pixel-independent scalar, so `Σ_k weight·sh` = `(Σ_k weight)·sh`. The per-ring
  inner loop collapses to two precomputed scalars per band (`Wfull[i]`,
  `Waniso[i]`); the float32 multiply-accumulate order on the rasters is preserved.
- **Host-side azimuth reads** (`SOLWEIG_FAST_ACCUM`, default on): the direction
  tests (`0 <= azimuth < 180`, …) previously compared 0-dim GPU tensors, costing
  ~1200 GPU→CPU syncs per tile; the azimuth table is now read to a host list once.
  float32 → Python float conversion is exact, so every branch decision is
  unchanged.
- `shadow()` itself: `bush_present` is evaluated once per patch instead of
  `bush.max() > 0.` per march step (one host sync instead of hundreds).

### 1.5 Shadow-matrix save time (`SOLWEIG_NPZ_COMPRESSLEVEL`, default `1`)

`np.savez_compressed` hardwires zlib level 6, which cost 10–25 s/tile on the ~2 GB
0/1 shadow-matrix trio. The arrays are now written through `zipfile.ZipFile` with
a configurable deflate level (default 1 — these binary masks compress nearly as
well at a fraction of the CPU cost). **Same keys, same `.npz` container, still
`np.load`-compatible**; `SOLWEIG_NPZ_COMPRESSLEVEL=default` restores
`np.savez_compressed` exactly. Cache round-tripping through
`utci_process.load_cached_svf_outputs` was verified bit-exact.

### 1.6 Removed `torch.cuda.empty_cache()` calls

Two calls in `shadow.py` (and two more in `solweig_gpu.py`, see §3) were removed.
The caching allocator reuses freed blocks for the next tile; `empty_cache()`
instead `cudaFree`s everything and forces expensive `cudaMalloc`s (+ device sync)
on the next tile. Peak memory is unaffected because the tensors are still `del`d.

### Measured result (per tile, `svf_calculator` end-to-end incl. saves)

| Tile | Original | Optimized | Speedup |
|---|---|---|---|
| 1020 × 1020 (1000 + 2·10 overlap) | ~34.5 s | **4.66 s** | 7.4× |
| 2020 × 2020 (2000 + 2·10 overlap) | ~422 s | **20.6 s** | 20.5× |

The speedup grows with tile size: the batched march keeps the GPU saturated
instead of being bounded by per-step launch/sync overhead.

---

## 2. Walls & aspect (`solweig_gpu/walls_aspect.py`)

### 2.1 `findwalls()` — scipy maximum filter

Wall height at each cell is *(max of its 4 orthogonal neighbours) − itself*, i.e.
a maximum filter over a plus-shaped footprint. The per-pixel Python double loop is
replaced with one `scipy.ndimage.maximum_filter(dem_array, footprint=domain,
mode='nearest') - dem_array` call (~400× faster at function level). Borders are
zeroed explicitly exactly as before; the interior is identical.

### 2.2 `filter1Goodwin_as_aspect_v3()` — exact two-stage vectorization

The original is a 180-angle loop that rotates two filters per angle and then
visits every wall pixel in Python. It is split into exact, vectorized stages
(signature unchanged):

1. **`_rotated_aspect_filters(scale)`** — builds all 180 rotated line/side filters
   once and caches them with `lru_cache` (they depend only on `scale`, not the
   DSM, so they are identical for every tile of a raster; the original recomputed
   360 `scipy.ndimage.rotate` calls per tile).
2. **`_winner_angle_map()`** — the winning angle per wall pixel via wall-count
   correlation: batched `F.conv2d` over angle chunks on the GPU (chunk size 16,
   halving backoff on OOM), or `scipy.ndimage.correlate` per angle on CPU. The
   correlation of a 0/1 mask with a 0/1 kernel is an exact small integer even in
   float32, so GPU and CPU produce identical winner maps, and a strict-`>` running
   update keeps the first (lowest-angle) winner exactly as the reference loop did.
3. **`_resolve_sides_exact()`** — the building-side decision (which way the wall
   faces) re-reads the DSM cells of the winning angle and reduces them with the
   reference's own float32 `np.sum` ordering (offsets enumerated in C order by
   `_side_offsets`). This step deliberately does *not* use conv/correlate: any
   other summation order rounds genuinely near-tied sides the other way and flips
   the aspect by 180°.

One incidental robustness change: the final fallback fill for wall pixels with no
winning angle now tests `(walls > 0)` on the wall-height raster — equivalent to
the original's `(walls == 1)` test on its internally binarized copy, but exact
regardless of dtype.

### 2.3 Orchestration & caching

- **GPU path**: when CUDA is available, tiles are processed sequentially
  in-process (`run_parallel_processing`) — a single process must own the device;
  multiple workers contending for one GPU is slower and can OOM. The CPU fallback
  keeps the original `ProcessPoolExecutor` fan-out.
- **Geometry cache reuse**: walls/aspect depend only on geometry, not on the
  simulation date. `run_parallel_processing(..., overwrite=False)` (new parameter,
  default) now skips tiles that already have both outputs and prints
  `Reusing N/M existing walls/aspect tiles`. `process_file_parallel`'s args tuple
  gained the `overwrite` element accordingly.

### Measured result

16 tiles (1020²): **63.6 s → 3.0 s (~21×)** end-to-end. Verified bit-identical on
two datasets with different derived filter sizes (including `filtersize=9`).

---

## 3. Correctness fixes (`solweig_gpu/utci_process.py`, `solweig_gpu/solweig_gpu.py`, `solweig_gpu/shadow.py`, `solweig_gpu/sun_position.py`)

1. **Date-aware metfile selection — new `map_metfiles_by_key()`** (used by
   `run_utci_tiles`). The original mapped *any* `.txt` in `metfiles/` by tile key,
   so when metfiles for several dates accumulated in one folder (the standard
   ERA5/WRF naming is `metfile_<i>_<j>_<YYYY-MM-DD>.txt`), a tile could silently
   run with **another date's** meteorology. The new mapper prefers the file whose
   date suffix matches `selected_date_str`, falls back to the date-less own-met
   form (`metfile_<i>_<j>.txt`), ignores other dates, and warns about tiles that
   only have metfiles for other dates.
2. **Single-timestep metfiles** — `np.loadtxt(..., ndmin=2)` in `run_utci_tiles`:
   a metfile with one row no longer collapses to a 1-D array (which crashed
   downstream indexing).
3. **`diffsh` vectorization** in `compute_utci`: the per-patch Python loop is
   replaced by the single elementwise expression
   `diffsh = shmat - (1 - vegshmat) * (1 - transVeg)` — bit-identical, no Python
   overhead over the 153 patches.
4. **`run_walls_aspect(preprocess_dir, overwrite=False)`** — forwards the new
   caching parameter (see §2.3); default behavior on a fresh directory is
   unchanged.
5. **Removed two `torch.cuda.empty_cache()` calls** (after each SVF tile in
   `calculate_svf`, after each tile in `run_utci_tiles`) — same rationale and
   effect as §1.6.
6. **`shadow()` bush branch: off-by-one slice (`solweig_gpu/shadow.py`, per-patch
   path)**. The bush line subtracted 1 from its slice start indices
   (`tempbush[xp1-1:xp2, yp1-1:yp2] = bush[xc1-1:xc2, yc1-1:yc2] - dz`), a leftover
   from the 1-based MATLAB original; the neighbouring `vegdem`/`vegdem2`/`a` lines do
   not. Every march step shifts along at least one axis, so a start index of 0 wrapped
   to -1 and the branch raised `RuntimeError: The expanded size of the tensor (H) must
   match the existing size (0)` on the first tile with `bush.max() > 0` (on a zero-shift
   axis it silently copied only the last row/column). The line now uses the same
   bounds as its neighbours, i.e. the MATLAB reference. Upstream crashed on the same
   input until v2.0.0, which fixed the line the same way; bush-free tiles never
   execute the branch and are unaffected. Regression
   tests: `tests/test_shadow_bush.py`.
7. **Sun position when `hours` keeps only some rows (`solweig_gpu/sun_position.py`)**.
   - Forcing rows are averages over the preceding step (ERA5 radiation is the mean
     over the hour ending at the timestamp), so `Solweig_2015a_metdata_noload` places
     the sun half a step before each row. It took that step from the gap between
     the first two rows.
   - `preprocess(hours=[4, 9, 12])` keeps rows 5 h apart, so the sun was placed
     2.5 h early. On Mumbai (IST) the 09:30 / 14:30 / 17:30 bands got the sun of
     07:00 / 12:00 / 15:00 (altitude 12° / 81° / 56° instead of 40° / 70° / 29°).
     A one-row metfile got no offset.
   - The observed effect on tile `0_10000`:
     - the 09:30 band was the most shaded;
     - sunlit UTCI was 47.2 °C at 09:30 but 38.5 °C at 14:30, although 14:30 had
       2 °C warmer air and twice the global radiation;
     - at 17:30 the diffuse/direct split left a direct beam of 29 W/m² instead
       of ~450.
   - Now the step is the smallest gap between rows, capped at 1 h because
     ERA5/WRF forcing is hourly. A kept hour gets exactly the sun of the same row
     in a full-day run.
   - Unchanged: contiguous hourly and 30-min metfiles, including every full-day
     run (bit-identical `altitude`/`azimuth`/`altmax`).
   - Changed, as intended: runs with gaps between rows, and one-row metfiles
     (now 30 min instead of 0).
   - Multi-hourly own-met files (documented as unsupported) are capped at a
     30-min offset.
   - The Mumbai figures in §4 *Albedo inputs* were measured with the old sun
     time. Their equivalence results still hold, but the absolute deltas will
     differ.
   - Tests: `TestSunTimeOffset` in `tests/test_sun_position.py`.
   - CTCM also processes one hour per call and so puts the sun at the row time
     itself (one-row branch). Comparisons with it differ by 30 min of sun
     position.
8. **Land-cover codes the solver sees on the GPU (`solweig_gpu/utci_process.py`)**.
   - `compute_utci` recodes land cover before use: 3/4 (trees) → 5 (grass), and
     codes outside 1–7 → 6 (bare soil). It did so on
     `lcgrid_np = lcgrid_torch.cpu().numpy()`.
   - On the GPU `.cpu()` makes a copy, so the solver still got the raw codes in
     `lcgrid_torch`. The solver treats code 3 as water: ground temperature set
     from the daily mean air temperature (`solweig.py:130`), and water Lup at
     night (`solweig.py:2129`). On the CPU the copy shares memory, so CPU runs
     already used the recoded grid, and GPU and CPU results differed wherever the
     land cover held code 3.
   - The package's own WorldCover recoding produces code 3 (`create_inputs.py`,
     shrubland and mangroves); OpenUrban-derived land cover does not.
   - The solver now gets `torch.from_numpy(lcgrid_np)`, so GPU runs match CPU
     runs, and UMEP, which rejects 3/4 outright.
   - Unchanged: runs without land cover, and land cover without code 3.
   - Test: `test_solver_gets_the_recoded_land_cover` in `tests/test_landcover.py`,
     which fails on the GPU without the fix.

---

## 4. New capability: ERA-5 forcing downloader

### `solweig_gpu/download_era5.py` (new, 514 lines)

Fetches exactly the two NetCDF files the preprocessor expects
(`data_stream-oper_stepType-instant.nc` with t2m/d2m/sp/u10/v10, and
`...-accum.nc` with ssrd/strd) via the CDS API:

- **Area from your rasters**: `domain_bbox_from_raster()` reprojects the Building
  DSM/DEM corners to EPSG:4326, pads by `--margin` (default 0.5°), and snaps
  outward to the ERA5 grid — the download always matches the modelled city.
- **Latest available date by default**: `latest_era5_date()` accounts for the
  ~5–6 day ERA5T publication lag, and the request automatically steps back up to
  `retry_back_days` if the newest date is not yet published.
- **Two sources**: `--source era5` (single-levels, 0.25°) or `--source era5-land`
  (0.1°). ERA5-Land accumulates `ssrd`/`strd` from 00 UTC with daily resets, so
  the helper de-accumulates them into per-hour totals to match single-levels
  (`_deaccumulate_era5_land`); output filenames and the rest of the workflow are
  identical, `data_source_type` stays `'ERA5'`.
- **Entry points**: `python -m solweig_gpu.download_era5 --dsm ... --out ...`, a
  `solweig_download_era5` console script (`setup.py`), a `download` extra
  (`pip install solweig-gpu[download]` → `cdsapi`), and a Python API
  `download_era5(dsm_path, out_folder, date=None, times=None, ...)`. On success it
  prints a ready-to-paste `thermal_comfort(...)` call.

`docs/input_data.md` gained a section documenting CDS account/licence setup and
this workflow (including the ERA5T real-time-lag warning and the ERA5-Land tip).

### Staged pipeline example (`run.py`, `processing.py`, `config.toml`)

The local driver was rewritten from a single `thermal_comfort(...)` call to the
staged API that the package exports: `download_era5` → `preprocess` →
`run_walls_aspect` → `calculate_svf` → `run_utci_tiles` → `mosaic_tiles` (new, in
`solweig_gpu/postprocess.py`).

- **Caching contract**, explained in comments: walls/aspect/SVF are geometry-only.
  They survive under `processed_inputs/{walls,aspect,SVF}` across runs, and are only
  valid while the input rasters, `tile_size` and `overlap` are unchanged. Change
  those → set `overwrite_geometry = true` or delete the cache dirs.
- Re-running for a different time window then costs seconds instead of re-running
  the whole geometry stack.
- The ERA5 step downloads the date's forcing from the CDS. It skips the download when
  the ERA5 folder already covers the time window, the DSM's area and the requested
  source's grid.
- **Files:**
  - `processing.py` holds the pipeline.
  - `config.toml` holds every path and setting. It is read with the standard
    library's `tomllib`, and `processing.load_config` rejects unknown, missing or
    mistyped settings.
  - `run.py` runs the pipeline with `config.toml`, or another config file given on
    the command line.
- `config.toml` holds the Mumbai settings; its `base_path` (`data/mumbai/`) is
  relative to the config file. Treat these three files as an example, not
  library code.

### Optional building clipping (`clip_buildings`)

Upstream `compute_utci` always sets building footprints (Building DSM − DEM
≥ 2 m, the existing `valid_mask`) to NaN in the UTCI output, with no way to
keep the rooftop values, and never masks the Shadow output. A new
`clip_buildings` argument (default `True`) on `compute_utci`, `run_utci_tiles`
and `thermal_comfort` (CLI `--clip_buildings`) now controls the
mask for **both** UTCI and Shadow:

- `True`: footprints are NaN in UTCI (as before) **and in Shadow (new default
  behavior)**. Pixels with no DSM/DEM data are masked too, as UTCI already was.
- `False`: both outputs keep the rooftop values.

Only the written arrays change. The `buildings` grid passed to the radiation
code and the `shadow` array used by WBGT stay unmasked, so the model itself
is unaffected. `torch.where(valid_mask, x, nan)` replaces the old `full(nan)` +
masked assignment, which it matches bit-for-bit.

The UTCI and Shadow GeoTIFFs record the setting in a dataset metadata item,
`CLIP_BUILDINGS=YES|NO`, which `utci_process.clip_state(path, var)` reads. It
returns `None` for an unreadable file. Untagged files were written before the
option existed, so they are read with the old behavior: UTCI clipped, Shadow not.
`run_utci_tiles(overwrite=False)` now also recomputes any tile whose UTCI or
(when `save_shadow`) Shadow tag differs from the requested setting, so flipping
the flag mid-run cannot silently mix clipped and unclipped tiles. Legacy tiles
that have a Shadow output match neither setting and are recomputed once.

Tests: `tests/test_clip_buildings.py`, which covers the tag reader, the resume
matrix, and an end-to-end run on a synthetic tile with a precomputed SVF cache.

### Albedo inputs (`albedo_filename`, `albedo_walls` / `albedo_ground`, cool-roof air-temperature cut)

Upstream uses a flat ground albedo of 0.15, or per-class values from
`landcoverclasses_2016a.txt` when a land-cover raster is given, and a fixed wall
albedo of 0.2. This matches how WRI's CTCM / cities-heat-resilient-infrastructure
workflow handles albedo:

- **Albedo raster**: `preprocess(..., albedo_filename=...)`, also on
  `thermal_comfort` and CLI `--albedo`. It takes fractions 0–1 on any
  grid.
  - If the grid differs from the Building DSM, it is resampled bilinearly onto it
    (`albedo_aligned.vrt`), then tiled into `processed_inputs/Albedo/`.
  - `compute_utci` uses it as the ground albedo grid `alb_grid`, overriding the
    land-cover or flat value everywhere, roofs included (as CTCM does).
  - NaN/nodata pixels keep the land-cover or flat value.
  - Values outside [0, 1] are rejected with a hint to divide percent rasters by 100.
- **`albedo_walls` / `albedo_ground`** on `compute_utci`, `run_utci_tiles` and
  `thermal_comfort` (CLI `--albedo_walls` / `--albedo_ground`) replace
  the fixed 0.2 / 0.15. The module constants `albedo_b` / `albedo_g` remain the
  defaults.
- **Cool-roof air-temperature cut**: with `baseline_albedo_filename`, `preprocess`
  computes the domain-mean albedo increase Δα. It writes
  `cut = 6·Δα` °C (Krayenhoff et al. 2021) to `processed_inputs/Albedo/air_temp_cut.json`.
  - `run_utci_tiles` applies `Ta −= cut × f(local hour)`, where `f` interpolates
    12:00 → 1, 15:00 → 0.935315 and 18:00 → 0.646853, and is 0 outside 12–18 h
    (warning printed).
  - The cut is applied to the forcing Ta before anything uses it, so SOLWEIG, WBGT
    and UTCI all see it, like CHRI's reduced met file. −999 gaps are left alone.
  - `run_utci_tiles(air_temp_cut=...)` overrides the recorded value.

Nothing in `solweig.py` changes. Albedo still only enters the reflected shortwave
terms (ground view factors → Kup; `albedo_walls` → Kdown/Kup/Kside wall
reflections), and surface temperatures still come from land cover.

The UTCI GeoTIFF records the settings in a new metadata item,
`ALBEDO=raster=<digest|none>;ground=<g>;walls=<g>;cut=<g>`, which
`albedo.albedo_state(path)` reads. Untagged outputs are read as the defaults
(`raster=none;ground=0.15;walls=0.2;cut=0`), so existing outputs are not
recomputed. `run_utci_tiles(overwrite=False)` recomputes a tile whose tag differs
from the requested settings. `preprocess` without an albedo raster removes an
`Albedo/` folder left by an earlier run, so a baseline run cannot pick up a
scenario's tiles or cut.

Tests: `tests/test_albedo.py` (32 tests) covers:
- a constant-0.15 raster and NaN/nodata gaps, both bit-identical to no raster;
- brighter ground (more Kup and Tmrt, same Kdown/Shadow);
- percent rejection, wall albedo, the hour factors, and the applied cut;
- the resume matrix;
- `preprocess` resampling, the recorded cut, and stale-folder removal.

Real data (Mumbai 2024-05-29, tile `0_2000`, 2020² at 1 m, 09:30/14:30/17:30 local):
- A constant-0.15 albedo raster is bit-identical to no raster for UTCI, Tmrt,
  Kup, Kdown, Shadow and Ta.
- Cool roofs (0.62 on DSM − DEM ≥ 2 m) with the matching cut (0.077 °C):
  - 09:30 (no cut): street Kup/UTCI exactly unchanged; rooftop UTCI +1.1 °C.
  - 14:30 / 17:30: Ta −0.073 / −0.053 °C as expected; street UTCI −0.11 / −0.08 °C;
    rooftop UTCI +2.2 / +1.1 °C.
  - Shadow unchanged in every band.
- Resuming with the same settings skips the tile; a changed albedo raster
  recomputes it.
- `preprocess` on a 600² crop (40% roof) with 10 m albedo rasters: Albedo tiles
  land on the DSM grid, and the recorded Δα is 0.18966 vs 0.18970 expected.

### Tree transmissivity (`tree_transmissivity_pct`)

`compute_utci` hard-codes a temperate deciduous season. On days 98–299 trees are in
leaf and 3% of the light gets through the canopy. Otherwise, about 26 October –
7 April, they are leafless and 50% of the direct sun gets through. That is wrong for
tropical cities, whose urban trees mostly keep their leaves, and the 3% cannot be
changed.

`tree_transmissivity_pct` on `run_utci_tiles` / `thermal_comfort` / `compute_utci`
sets the percent (0–100) of the light that gets through tree canopies on every
date. Like UMEP's "Transmissivity of light through vegetation (%)" (`TRANS_VEG`)
and CTCM's `transmissivity_of_light_through_vegetation`, it applies to both the
direct sun (`psi`) and the diffuse sky light (`svfbuveg`, `diffsh`). It replaces
this fork's earlier `evergreen_trees` switch and its default, the module constant
`conifer_bool`.

- The default (`None`) is bit-identical to the hard-coded season.
- `3` keeps trees in leaf all year. It is bit-identical to the old
  `conifer_bool = True` path.
- `50` is not the old leafless canopy, which kept 3% for the diffuse light.
- Values outside 0–100, NaN and `True`/`False` raise `ValueError`.
- UTCI and Shadow record the setting in a `TREE_TRANSMISSIVITY_PCT` metadata item:
  `seasonal` for `None`, otherwise the percentage, e.g. `3`.
- When a run resumes with `overwrite=False`, a tile whose UTCI or Shadow was
  written with another value is recomputed. Untagged outputs are recomputed once,
  since their trees may have been seasonal or in leaf all year.
- The Indian-city runs pass `tree_transmissivity_pct=3`.
- `SkyViewFactor_*.tif`, the total SVF that `calculate_svf` caches, keeps 3%. It
  is geometry-only, and the radiation calculation does not use it.
- Verification: a synthetic tile with a tree and canopy in the sky view was run in
  and out of the leaf season. UTCI, Tmrt, Kdown, Kup and Shadow were bit-identical
  in both pairings: `None` against the hard-coded season, and `3` against the
  in-leaf path.
- Tests: `tests/test_tree_transmissivity.py`.
  - With the season moved off the metfile's day, tree shade is 0.5 by default,
    0.03 at 3% and 0.25 at 25%. Every other Shadow pixel is identical.
  - With half of the sky seen through canopy and no tree shadow, Kdown rises at
    50%, and the default matches 3% bit for bit.

### Land cover: OpenUrban conversion and code check (`solweig_gpu/landcover.py`, new)

`compute_utci` accepts the SOLWEIG land-cover codes 1–7 and turns every other code
into 6 (bare soil), with a per-tile warning. WRI OpenUrban land cover (codes
110–622) passed as `landcover_filename` therefore ran as bare soil everywhere.

- `python -m solweig_gpu.landcover OpenUrban.tif Landcover.tif`
  (`openurban_to_solweig`) converts it.
  - It uses CIF's `city_metrix.layers.open_urban.reclass_map`, the mapping CTCM
    applies: roads, parking and other built-up → 1, buildings → 2, green and
    public open space → 5, barren → 6, water → 7.
  - It writes a Byte GeoTIFF (tiled, DEFLATE) on the same grid. NaN, nodata
    and unknown codes become 0.
  - Bengaluru (35032×34257 px) converts in 22 s.
- `preprocess` now calls `check_landcover_codes`. It samples 64 rows and stops the
  run if more than half of the valid pixels are outside 1–7. When the codes look
  like OpenUrban, the message points at the converter.
- Without a land cover (or wind coefficient) raster, `preprocess` removes a
  `Landcover/` (`WindCoeff/`) folder left by an earlier run, as it already did
  for `Albedo/`. `run_utci_tiles` uses these layers whenever their folder exists.
- `docs/input_data.md` listed wrong codes (3 water, 4 vegetation, 5 bare soil). It
  now lists the codes the model uses.
- Tests: `tests/test_landcover.py` covers:
  - the mapping, including NaN and unknown codes;
  - the code check and its use in `preprocess`;
  - the stale folder;
  - the solver fix in §3.8.

Real data: a Bengaluru 2000×2000 crop at (12000, 16000), with 45% paved, 27%
buildings and 28% grass, run with a 3-row own met file. Figures are for the
14:30 band.
- Shadow is identical with and without land cover and albedo.
- Land cover vs none, on sunlit ground: Lup +75 W/m² on paved, −84 W/m² on
  grass, −125 W/m² on water.
- Adding the albedo raster leaves Lup unchanged. It raises Kup on sunlit ground
  by 19 W/m² (median), correlating 0.97 with the albedo raster minus the
  land-cover albedo.
- UTCI is NaN exactly on DSM − DEM ≥ 2 m buildings in every configuration.

### Packaging (`requirements.txt`, `requirements-lock.txt`)

- `numpy==2.3.5`, `pandas==3.0.3` — pandas 3.0.4's cp314 wheel segfaults on
  `Timedelta` arithmetic under Python 3.14; 3.0.3 is fine.
- `requirements-lock.txt` (new): a fully pinned, mutually consistent environment
  for Python 3.14. torch and its CUDA stack are intentionally omitted — install
  torch from the appropriate wheel index first (tested with 2.12.1+cu130).

---

## 5. Verification: how "unchanged output" was established

Everything above is backed by `tools/ab_harness/` (new), which runs the
**original** and **current** package in separate processes against the same tiles
(`--version cur` imports this checkout and `--version orig` the reference tree at
`--orig-root`; each puts its root first on `sys.path` and asserts
`solweig_gpu.__file__` before running) and compares results:

| Script | Purpose |
|---|---|
| `make_synthetic.py` | write a deterministic 4-tile `processed_inputs` folder of a synthetic city, so the harness runs without private data; `--bush` adds phantom-bush pixels that exercise the per-patch `shadow()` path |
| `dump_svf.py` | run `svf_calculator` on one tile, dump all 19 returned arrays losslessly + timing JSON; `--artifacts` also exercises the `save_rasters=True` path |
| `compare_svf.py` | compare two dumps; shadow matrices must be bit-equal; `svf*` at float32-reassociation tolerance; `strict` mode requires all 19 bit-equal (used for flag-toggle A/B) |
| `dump_walls.py` / `compare_walls.py` | time `run_parallel_processing` / per-function `findwalls` + aspect; bitwise-compare `walls_*`/`aspect_*` GeoTIFFs incl. geo metadata |
| `compare_artifacts.py` | compare saved artifact sets (SkyViewFactor tif, `svfs.zip` members, shadow-matrix npz members) |
| `validate_lsb.py` | rebuild every `svf*` array in float64 from the (bit-identical) shadow matrices and measure each version's distance from that float64 truth, under the svfN rule each version was made with |
| `profile_svf.py` | per-altitude-band march timings and march/accumulate/save split |
| `roundtrip.py` | load a written SVF cache via `utci_process.load_cached_svf_outputs` and check it matches the in-memory dump bit-for-bit |

**Findings (2026-07-02):**

- **Walls, aspect, and the three shadow/vegetation-shadow matrices are
  bit-identical** to the original, on every tile tested (two datasets, multiple
  tile sizes).
- The **15 `svf*` rasters differ only by float32 summation-order rounding** (max
  observed ~4e-6): the accumulation `Σ (weight·sh)` was reassociated to
  `(Σ weight)·sh` (§1.4). `validate_lsb.py` shows the optimized order is *closer*
  to the float64 ground truth than the original's.
- **Every optimization flag was additionally A/B'd against the optimized baseline
  in `strict` (all-19-arrays bit-equal) mode**, i.e. compiled vs eager, batched vs
  per-patch, schedule vs stepwise, fast-freeze vs 4-way freeze, march ceiling on
  vs off all produce bit-identical tensors.
- Cache round-trip (npz at deflate level 1) is bit-exact through
  `load_cached_svf_outputs`.

### Three pre-existing quirks (deliberately left untouched)

Found while validating; all reproduce identically in the original and are
**preserved** for bit-compatibility, but upstream may want to fix them properly
(each would slightly change outputs):

1. **`iazimuth` float32 truncation** (`svf_calculator`): the azimuth table is
   built with `int(360 / interval)` where `interval` is a float32 quotient;
   truncation gives four bands one entry fewer than nominal (30→29, 28→27, 13→12,
   7→6 for `patch_option=2`), so the table misaligns against the nominal bands and
   the last 4 of 153 slots keep azimuth 0.0.
2. **Zenith patch marches to the grid bound**: for the altitude-90° band, float32
   `tan(90°)` is *negative*, so its `dz` never exceeds the ceiling and the march
   runs ~`max(H, W)` steps doing nothing useful. (This is also why the optional
   batch padding `SOLWEIG_BATCH_PAD` defaults off — padding that 1-lane batch
   would multiply the waste.)
3. **Negative DEM values create phantom "bushes"**: `bush` is derived from absolute
   elevations, `bush = logical_not(vegdem2_height * vegdem_height) * vegdem_height`
   (`svf_calculator`; the same formula is in `compute_utci`). With the built-in 25 %
   trunk zone there can be no bushes by construction, but wherever the DEM is negative
   and `0.25 * trees + DEM == 0` exactly (e.g. trees 3.0 m over DEM -0.75 m) the product
   is 0 and the pixel becomes a 2.25 m "bush". On the Mumbai dataset this hits a
   handful of pixels on 2 of 231 tiles (`4000_36000`, `4000_38000`); those tiles are
   routed to the slower per-patch `shadow()` path (which is where the §3.6 crash
   surfaced) and the affected pixels get the bush treatment in both the SVF and UTCI
   stages. Deriving `bush` from relative heights would remove the artefact but changes
   outputs, so it is left as is.

---

## 6. Tuning & compatibility reference

### Environment flags (all optional; defaults = the fast paths)

Every optimization can be disabled independently, which restores the original
code path bit-for-bit — useful for A/B benchmarking or as an escape hatch.

| Variable | Default | Meaning (value to restore original behavior) |
|---|---|---|
| `SOLWEIG_BATCH_MARCH` | `1` | batched multi-azimuth march in `svf_calculator` (`0` → per-patch `shadow()`) |
| `SOLWEIG_BATCH_SIZE` | `16` | azimuth lanes per batch |
| `SOLWEIG_BATCH_CLAMP` | `1` | VRAM-aware clamp of the batch size (`0` → use `SOLWEIG_BATCH_SIZE` verbatim) |
| `SOLWEIG_BATCH_SCHEDULE` | `1` | precomputed march schedule, one host sync per batch (`0` → stepwise loop) |
| `SOLWEIG_BATCH_PAD` | `0` | pad tail batches to a fixed lane count for one compiled shape per tile (off: see zenith quirk) |
| `SOLWEIG_FAST_FREEZE` | `1` | mask only the accumulator when `a.min() >= 0` (`0` → 4-way freeze) |
| `SOLWEIG_FAST_ACCUM` | `1` | host-side azimuth direction tests (`0` → tensor comparisons) |
| `SOLWEIG_MARCH_CEILING` | `1` | march to the surface height *range* (`0` → original absolute ceiling) |
| `SOLWEIG_COMPILE_MARCH` | `1` | `torch.compile` the per-patch march core (CUDA only; `0` → eager) |
| `SOLWEIG_COMPILE_BATCH` | `1` | `torch.compile` the batched march step (CUDA only; `0` → eager) |
| `SOLWEIG_NPZ_COMPRESSLEVEL` | `1` | deflate level for the shadow-matrix npz (`default` → `np.savez_compressed`, i.e. level 6) |

### API surface

- Package exports (`solweig_gpu/__init__.py`): one addition, `mosaic_tiles` (new
  `solweig_gpu/postprocess.py`).
- Backward-compatible signature extensions: `run_walls_aspect(preprocess_dir,
  overwrite=False)`, `run_parallel_processing(..., overwrite=False)`,
  `compute_utci(..., clip_buildings=True)`, `run_utci_tiles(..., clip_buildings=True)`,
  `thermal_comfort(..., clip_buildings=True)`, and the CLI flag `--clip_buildings`.
- Albedo extensions (all keyword-only additions at the end of each signature):
  - `preprocess(..., albedo_filename=None, baseline_albedo_filename=None)` and
    `ppr(..., albedo_filename=None, baseline_albedo_filename=None)`;
  - `compute_utci(..., albedo_path=None, albedo_walls=0.2, albedo_ground=0.15, air_temp_cut=0.0)`;
  - `run_utci_tiles(..., albedo_walls=0.2, albedo_ground=0.15, air_temp_cut=None)`;
  - `thermal_comfort(..., albedo_filename=None, baseline_albedo_filename=None,
    albedo_walls=0.2, albedo_ground=0.15)`;
  - CLI flags `--albedo`, `--baseline_albedo`, `--albedo_walls`, `--albedo_ground`.
- Tree transmissivity: `compute_utci(..., tree_transmissivity_pct=None)`,
  `run_utci_tiles(..., tree_transmissivity_pct=None)`,
  `thermal_comfort(..., tree_transmissivity_pct=None)` (Python API only; no CLI flag).
- Land cover: the new `solweig_gpu.landcover` module (`openurban_to_solweig`,
  `check_landcover_codes`, `OPENURBAN_TO_SOLWEIG`; runnable with `python -m`).
  `preprocess` / `ppr` signatures are unchanged.
- `process_file_parallel` (internal worker) takes a 5-tuple (added `overwrite`).
- New public names: `shadow_batch` (`shadow.py`), `map_metfiles_by_key`,
  `clip_state`, `CLIP_TAG`, `check_tree_transmissivity`, `tree_state`, `tree_tag`
  and `TREES_TAG` (`utci_process.py`), the `solweig_gpu.download_era5`
  module, the `solweig_download_era5` console script, and the `solweig_gpu.albedo`
  module (`ALBEDO_TAG`, `albedo_state`, `albedo_tag`, `air_temp_cut_factor`, ...).
- CPU-only installs keep working: the `torch.compile` wrappers and the GPU
  walls/aspect branch are gated on `torch.cuda.is_available()`; scipy paths cover
  the CPU case.
- File formats: GeoTIFF writing is unchanged except that the UTCI and Shadow outputs
  carry `CLIP_BUILDINGS=YES|NO` and `TREE_TRANSMISSIVITY_PCT=...` dataset metadata
  items, UTCI also carries an `ALBEDO=...` item, and with the default
  `clip_buildings=True` Shadow is NaN on building footprints (§4). The
  shadow-matrix `.npz` keeps the same keys and stays `np.load`-compatible (only
  the deflate level changes).
- First-tile latency: `torch.compile` compiles the march graphs once per process
  (a few seconds), then replays them for every subsequent step/tile; compile
  failures fall back to eager automatically (`suppress_errors=True`).

### Not part of the proposed change set

`data/` contents and the `.claude/` / `.pytest_cache/` working artifacts are
local to this fork's development environment.

---

## 7. Reproducing the verification & benchmarks

```bash
# from the fork root, with any Python that has the package's requirements
V=python
git worktree add ../solweig-ref 38b451d        # the baseline these numbers were measured against
D=/tmp/ab_data; O=/tmp/ab; mkdir -p $O
$V tools/ab_harness/make_synthetic.py $D       # or --data any processed_inputs folder

# SVF: dump both versions on one tile, then compare
$V tools/ab_harness/dump_svf.py --version orig --orig-root ../solweig-ref --data $D --tile 0_0 --out $O/orig.npz --artifacts $O/art_orig
$V tools/ab_harness/dump_svf.py --version cur --data $D --tile 0_0 --out $O/cur.npz --artifacts $O/art_cur
$V tools/ab_harness/compare_svf.py $O/orig.npz $O/cur.npz orig-vs-cur     # LSB tolerance for svf*
$V tools/ab_harness/validate_lsb.py $O/orig.npz $O/cur.npz 0_0 $D         # float64-truth distance
$V tools/ab_harness/compare_artifacts.py $O/art_orig $O/art_cur 0_0       # saved SVF cache files
$V tools/ab_harness/roundtrip.py $O/art_cur 0_0 $O/cur.npz $D             # npz cache round-trip

# flag-toggle A/B (bit-equality required):
SOLWEIG_COMPILE_BATCH=0 $V tools/ab_harness/dump_svf.py --version cur --data $D --tile 0_0 --out $O/eager.npz
$V tools/ab_harness/compare_svf.py $O/cur.npz $O/eager.npz compiled-vs-eager strict

# walls/aspect: native timing + bitwise GeoTIFF comparison
$V tools/ab_harness/dump_walls.py --version orig --orig-root ../solweig-ref --mode native --data $D --out $O/w_orig
$V tools/ab_harness/dump_walls.py --version cur --mode native --data $D --out $O/w_cur
$V tools/ab_harness/compare_walls.py $O/w_orig $O/w_cur

git worktree remove ../solweig-ref
```

Against `38b451d`, `svfN`, `svfNveg` and `svfNaveg` are reported as expected
differences: upstream fixed their north test in v2.0.0 (see the merge note at the
top). Use `origin/main` as the reference to check the merged code against v2.0.0.
See `tools/ab_harness/README.md` for every script and option.

---

## 8. Suggested review order

If upstreaming incrementally, the changes decompose into independent pieces, in
increasing order of review effort:

1. `utci_process.py` fixes — `map_metfiles_by_key`, `ndmin=2`, `diffsh` (small,
   standalone, fixes real failure modes).
2. `findwalls` → `maximum_filter` (6 lines of logic, bit-identical, no new deps —
   scipy is already required).
3. npz compression level + `empty_cache()` removals (small, benchmark-backed).
4. Walls/aspect rewrite + geometry-cache `overwrite` params.
5. SVF batched/compiled march (`shadow.py`) — the largest piece; every behavior is
   flag-gated with an original-path escape hatch, and `tools/ab_harness/` ships
   alongside to re-verify on the maintainer's hardware.
6. ERA-5 downloader + docs + packaging extras (independent new feature).
7. `clip_buildings` option (small; changes the default Shadow output, so it needs
   an upstream decision on the default).
