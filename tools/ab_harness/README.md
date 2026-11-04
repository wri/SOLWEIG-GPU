# A/B verification & benchmark harness (current vs original solweig_gpu)

Scripts used to verify that the optimized `solweig_gpu` produces the same output
as the original reference tree and to benchmark SVF / walls / aspect. Each run
happens in its own process; `--version orig` prepends the original repo root to
`sys.path` (which wins over the editable install) and asserts
`solweig_gpu.__file__` before doing anything.

Default paths point at the original tree at
`/home/arsh/wri/code/SOLWEIG-GPU-main_1/SOLWEIG-GPU-main` and the test1 dataset;
override with the CLI flags.

| script | purpose |
|---|---|
| `dump_svf.py` | run `svf_calculator` on one tile, dump all 19 returned arrays (lossless npz) + timing/meta JSON. `--artifacts DIR` also exercises the `save_rasters=True` path. |
| `dump_walls.py` | `--native`: time a full `run_parallel_processing`; `--direct`: per-tile `findwalls` / `filter1Goodwin_as_aspect_v3` timings + raw dumps. |
| `compare_svf.py` | compare two svf dumps. Shadow mats must be bit-equal; `svf*` pass at float32-reassociation tolerance. 4th arg `strict` requires all 19 bit-equal (for current-vs-current flag toggles). |
| `compare_walls.py` | bitwise-compare `walls_*/aspect_*` GeoTIFF pairs incl. geo metadata. |
| `compare_artifacts.py` | compare the saved artifact set (SkyViewFactor tif, svfs zip members, shadowmats npz members). |
| `validate_lsb.py` | rebuild every `svf*` array in float64 from the (bit-identical) shadow mats and measure each version's distance from the float64 truth — proves observed diffs are pure float32 summation-order rounding. Replicates the code's exact `iazimuth` table, including its float32 truncation quirk. |
| `profile_svf.py` | per-band `shadow_batch` timings, march/accumulation/save split, optional kernel profile of one batch. |
| `roundtrip.py` | load a written SVF cache through `utci_process.load_cached_svf_outputs` and check it matches the in-memory dump bit-for-bit. |

Typical flow:

```bash
V=.venv/bin/python
$V dump_svf.py --version orig --tile 0_1000 --out orig.npz
$V dump_svf.py --version cur  --tile 0_1000 --out cur.npz
$V compare_svf.py orig.npz cur.npz label            # cross-version (LSB tolerance)
$V compare_svf.py curA.npz curB.npz label strict    # flag toggle (bit-equal)
```

Findings as of 2026-07-02 (see session report): walls, aspect and the three
shadow matrices are bit-identical to the original; the 15 `svf*` rasters differ
only by float32 summation-order rounding (max ~4e-6, and the optimized order is
closer to the float64 truth than the original's).
