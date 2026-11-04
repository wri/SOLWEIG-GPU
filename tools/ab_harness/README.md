# A/B verification & benchmark harness (current vs reference solweig_gpu)

Scripts used to verify that the optimized `solweig_gpu` produces the same output
as a reference tree, and to benchmark SVF / walls / aspect.

**Version isolation.** `--version cur` imports `solweig_gpu` from the checkout the
harness is in; `--version orig` imports it from `--orig-root`. Either way that root
goes first on `sys.path` and `solweig_gpu.__file__` is asserted before anything
runs, so nothing needs to be installed and an installed copy cannot leak in. Each
run is its own process. Any Python environment with the package's requirements
will do; without CUDA everything runs on the CPU.

**Reference tree.** Any other checkout, typically a git worktree:

- `origin/main` (upstream v2.0.0): every check should pass.
- `38b451d` (the June 2026 baseline that `UPSTREAM_CHANGES.md` was measured
  against): `svfN`, `svfNveg` and `svfNaveg` are reported as expected
  differences. Before v2.0.0 the north test in `svf_calculator`
  (`270 <= azimuth < 90`) was never true, so the three arrays stayed zero; the
  scripts detect that and say so. This tree crashes on `--bush` tiles
  (`UPSTREAM_CHANGES.md` §3.6).

**Data.** `--data` is a `processed_inputs` folder with `Building_DSM/`, `DEM/` and
`Trees/` tiles named `<name>_<key>.tif`, as `preprocess` writes them.
`make_synthetic.py OUT` writes a small deterministic one, so the harness runs
without private data.

| script | purpose |
|---|---|
| `make_synthetic.py` | write a `processed_inputs` folder: 4 tiles (`0_0`, `0_200`, `200_0`, `200_200`) of a synthetic 400 × 400 px city at 2 m, tiled like `preprocess(tile_size=200, overlap=20)`. `--bush` adds phantom-bush pixels (§5 quirk 3) that send SVF down the per-patch `shadow()` path. |
| `dump_svf.py` | run `svf_calculator` on one tile, dump all 19 returned arrays (lossless npz) + timing/meta JSON (root, git revision, device, `SOLWEIG_*` flags). `--artifacts DIR` also exercises the `save_rasters=True` path. |
| `dump_walls.py` | `--mode native`: time a full `run_parallel_processing`; `--mode direct`: per-tile `findwalls` / `filter1Goodwin_as_aspect_v3` timings + raw dumps (all tiles unless `--tiles`). |
| `compare_svf.py` | compare two svf dumps. Shadow mats must be bit-equal; `svf*` pass at float32-reassociation tolerance. 4th arg `strict` requires all 19 bit-equal (for current-vs-current flag toggles). |
| `compare_walls.py` | bitwise-compare `walls_*/aspect_*` GeoTIFF pairs incl. geo metadata. |
| `compare_artifacts.py` | compare the saved artifact set (SkyViewFactor tif, svfs zip members, shadowmats npz members). |
| `validate_lsb.py` | rebuild every `svf*` array in float64 from the (bit-identical) shadow mats and measure each version's distance from the float64 truth — proves observed diffs are pure float32 summation-order rounding. Replicates the code's exact `iazimuth` table, including its float32 truncation quirk, and checks each dump against the svfN rule it was made with. |
| `profile_svf.py` | per-band `shadow_batch` timings, march/accumulation/save split, optional kernel profile of one batch (this checkout only). |
| `roundtrip.py` | load a written SVF cache through `utci_process.load_cached_svf_outputs` and check it matches the in-memory dump bit-for-bit. |
| `_common.py` | shared helpers: version isolation, CUDA-optional timing, the 19-name return order. |

The compare and validate scripts print a `RESULT`/`PASS` line and exit 1 on FAIL.

Typical flow, from the repo root:

```bash
V=python                                       # any env with the package's requirements
git worktree add ../solweig-ref origin/main    # reference; 38b451d reproduces UPSTREAM_CHANGES.md
D=/tmp/ab_data; O=/tmp/ab; mkdir -p $O
$V tools/ab_harness/make_synthetic.py $D

# SVF: dump both versions on one tile, then compare
$V tools/ab_harness/dump_svf.py --version orig --orig-root ../solweig-ref --data $D --tile 0_0 --out $O/orig.npz --artifacts $O/art_orig
$V tools/ab_harness/dump_svf.py --version cur --data $D --tile 0_0 --out $O/cur.npz --artifacts $O/art_cur
$V tools/ab_harness/compare_svf.py $O/orig.npz $O/cur.npz orig-vs-cur     # LSB tolerance for svf*
$V tools/ab_harness/validate_lsb.py $O/orig.npz $O/cur.npz 0_0 $D         # float64-truth distance
$V tools/ab_harness/compare_artifacts.py $O/art_orig $O/art_cur 0_0       # saved SVF cache files
$V tools/ab_harness/roundtrip.py $O/art_cur 0_0 $O/cur.npz $D             # cache loads back bit-exact

# flag toggle (all 19 arrays bit-equal)
SOLWEIG_COMPILE_BATCH=0 $V tools/ab_harness/dump_svf.py --version cur --data $D --tile 0_0 --out $O/eager.npz
$V tools/ab_harness/compare_svf.py $O/cur.npz $O/eager.npz compiled-vs-eager strict

# walls/aspect: native timing + bitwise GeoTIFF comparison
$V tools/ab_harness/dump_walls.py --version orig --orig-root ../solweig-ref --mode native --data $D --out $O/w_orig
$V tools/ab_harness/dump_walls.py --version cur --mode native --data $D --out $O/w_cur
$V tools/ab_harness/compare_walls.py $O/w_orig $O/w_cur

git worktree remove ../solweig-ref
```

Findings:

- 2026-07-02, against `38b451d` on two datasets that are not in the repo: walls,
  aspect and the three shadow matrices are bit-identical to the original; the 15
  `svf*` rasters differ only by float32 summation-order rounding (max ~4e-6, and
  the optimized order is closer to the float64 truth than the original's).
- 2026-11-04, after the merge with upstream v2.0.0, on the synthetic tiles: the
  same holds against `1fec79a` (v2.0.0), including `svfN*` (max 8.3e-7) and a
  `--bush` tile; against `38b451d`, only `svfN*` differ beyond rounding, by design.
  Compiled vs eager is bit-identical; CPU vs GPU has bit-identical shadow
  matrices and `svf*` within 2.4e-7 (2 float32 ulps at 1.0).
