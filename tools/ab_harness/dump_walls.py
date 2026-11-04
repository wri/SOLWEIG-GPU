#!/usr/bin/env python
"""Walls/aspect A/B driver.
--mode native : one full run_parallel_processing over the Building_DSM dir (wall-clock).
--mode direct : per-tile findwalls + filter1Goodwin_as_aspect_v3 with separate timers,
                dumping raw-dtype results (harness-only, lossless).
Version isolation: --version cur imports solweig_gpu from this checkout,
--version orig from --orig-root (see _common.py).
MUST stay __main__-guarded: the original path spawns a forkserver ProcessPool that
re-imports this module.
"""
import argparse
import inspect
import json
import os
import sys
import time

from _common import add_version_args, savez_fast, use_version


def main():
    ap = argparse.ArgumentParser()
    add_version_args(ap)
    ap.add_argument("--mode", choices=["native", "direct"], required=True)
    ap.add_argument("--data", required=True,
                    help="processed_inputs folder with a Building_DSM/ tile folder")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--tiles", default=None,
                    help="comma-separated tile keys for --mode direct (default: all tiles)")
    args = ap.parse_args()

    solweig_gpu, root, rev = use_version(args.version, args.orig_root)
    bdsm_dir = os.path.join(args.data, "Building_DSM")
    if not os.path.isdir(bdsm_dir):
        sys.exit(f"no Building_DSM folder in {args.data}")

    import numpy as np
    from osgeo import gdal
    gdal.UseExceptions()
    from solweig_gpu.walls_aspect import (run_parallel_processing, findwalls,
                                          filter1Goodwin_as_aspect_v3, walllimit)

    os.makedirs(args.out, exist_ok=True)
    meta = {"version": args.version, "mode": args.mode,
            "module": os.path.realpath(solweig_gpu.__file__), "root": root, "rev": rev,
            "data": os.path.abspath(args.data)}

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
        prefix = "Building_DSM_"
        tiles = args.tiles.split(",") if args.tiles else sorted(
            f[len(prefix):-len(".tif")] for f in os.listdir(bdsm_dir)
            if f.startswith(prefix) and f.endswith(".tif"))
        meta["tiles"] = {}
        for key in tiles:
            path = os.path.join(bdsm_dir, f"{prefix}{key}.tif")
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
