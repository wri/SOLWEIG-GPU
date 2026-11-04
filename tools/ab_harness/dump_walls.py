#!/usr/bin/env python
"""Walls/aspect A/B driver.
--native : one full run_parallel_processing over the Building_DSM dir (wall-clock).
--direct : per-tile findwalls + filter1Goodwin_as_aspect_v3 with separate timers,
           dumping raw-dtype results (harness-only, lossless).
MUST stay __main__-guarded: the original path spawns a forkserver ProcessPool that
re-imports this module.
"""
import argparse
import inspect
import json
import os
import sys
import time
import zipfile

ORIG_ROOT = "/home/arsh/wri/code/SOLWEIG-GPU-main_1/SOLWEIG-GPU-main"
CUR_ROOT = "/home/arsh/wri/code/SOLWEIG-GPU-main"
DEFAULT_DATA = os.path.join(CUR_ROOT, "data/test1/processed_inputs")


def savez_fast(path, **arrays):
    import numpy as np
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for name, arr in arrays.items():
            with zf.open(name + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(arr), allow_pickle=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["orig", "cur"], required=True)
    ap.add_argument("--mode", choices=["native", "direct"], required=True)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--tiles", default=None, help="comma keys for --direct")
    args = ap.parse_args()

    if args.version == "orig":
        sys.path.insert(0, ORIG_ROOT)
    import solweig_gpu
    expected = {"orig": ORIG_ROOT, "cur": CUR_ROOT}[args.version]
    actual = os.path.realpath(solweig_gpu.__file__)
    assert actual.startswith(os.path.realpath(expected) + os.sep), \
        f"import resolved to {actual}, expected under {expected}"

    import numpy as np
    from osgeo import gdal
    gdal.UseExceptions()
    from solweig_gpu.walls_aspect import (run_parallel_processing, findwalls,
                                          filter1Goodwin_as_aspect_v3, walllimit)

    bdsm_dir = os.path.join(args.data, "Building_DSM")
    os.makedirs(args.out, exist_ok=True)
    meta = {"version": args.version, "mode": args.mode, "module": actual,
            "data": args.data}

    if args.mode == "native":
        wout = os.path.join(args.out, "walls")
        aout = os.path.join(args.out, "aspect")
        kwargs = {}
        if "overwrite" in inspect.signature(run_parallel_processing).parameters:
            kwargs["overwrite"] = True  # required: current version silently reuses otherwise
        t0 = time.perf_counter()
        run_parallel_processing(bdsm_dir, wout, aout, **kwargs)
        meta["wall_clock_s"] = time.perf_counter() - t0
        meta["n_tiles"] = len([f for f in os.listdir(bdsm_dir) if f.endswith(".tif")])
    else:
        tiles = args.tiles.split(",")
        meta["tiles"] = {}
        for key in tiles:
            path = os.path.join(bdsm_dir, f"Building_DSM_{key}.tif")
            ds = gdal.Open(path)
            gt = ds.GetGeoTransform()
            scale = 1 / gt[1]
            a = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            ds = None
            t0 = time.perf_counter()
            walls = findwalls(a, walllimit)
            t1 = time.perf_counter()
            aspects = filter1Goodwin_as_aspect_v3(walls, scale, a)
            t2 = time.perf_counter()
            savez_fast(os.path.join(args.out, f"wa_{key}.npz"),
                       walls_raw=walls, aspect_raw=aspects,
                       walls_f32=walls.astype(np.float32),
                       aspect_f32=aspects.astype(np.float32))
            meta["tiles"][key] = {"t_findwalls_s": t1 - t0, "t_aspect_s": t2 - t1,
                                  "shape": list(a.shape), "scale": scale,
                                  "walls_dtype": str(walls.dtype),
                                  "aspect_dtype": str(aspects.dtype)}
            print(f"[{args.version}/{key}] findwalls {t1-t0:.2f}s aspect {t2-t1:.2f}s",
                  flush=True)

    with open(os.path.join(args.out, f"meta_{args.mode}.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
