#!/usr/bin/env python
"""Run svf_calculator for one tile with one package version; dump all 19 returned
arrays (lossless) + timing/meta JSON. Version isolation: --version cur imports
solweig_gpu from this checkout, --version orig from --orig-root (see _common.py).
"""
import argparse
import json
import os
import time

from _common import (NAMES, add_version_args, max_mem_gb, require_files, savez_fast,
                     sync, tile_paths, use_version)


def main():
    ap = argparse.ArgumentParser()
    add_version_args(ap)
    ap.add_argument("--tile", required=True)
    ap.add_argument("--data", required=True,
                    help="processed_inputs folder with Building_DSM/, Trees/ and DEM/ tiles")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--out", required=True, help="output .npz path (meta at .json)")
    ap.add_argument("--no-dump", action="store_true", help="timing only")
    ap.add_argument("--artifacts", default=None,
                    help="also run save_rasters=True standalone mode into this dir")
    args = ap.parse_args()

    solweig_gpu, root, rev = use_version(args.version, args.orig_root)
    bdsm, tree, dem = tile_paths(args.data, args.tile)
    require_files(bdsm, tree, dem)

    import torch
    from solweig_gpu.shadow import svf_calculator, load_raster_to_tensor

    # ---- Input prep: verbatim replica of svf_calculator's standalone mode (its
    # `if save_rasters:` branch; identical in this tree, v2.0.0 and 38b451d).
    a, gdal_dsm_local = load_raster_to_tensor(bdsm)
    temp1, _ = load_raster_to_tensor(tree)
    temp2, _ = load_raster_to_tensor(dem)
    geotransform = gdal_dsm_local.GetGeoTransform()
    scale = 1.0 / geotransform[1]
    temp1[temp1 < 0.0] = 0.0
    vegdem_height = temp1 + temp2
    vegdem2_height = torch.add(temp1 * 0.25, temp2)
    bush = torch.logical_not(vegdem2_height * vegdem_height) * vegdem_height
    vegdem = temp1 + a
    vegdem[vegdem == a] = 0.0
    vegdem2 = temp1 * 0.25 + a
    vegdem2[vegdem2 == a] = 0.0
    amaxvalue = torch.maximum(a.max(), vegdem_height.max())
    # ----

    meta = {
        "version": args.version, "tile": args.tile,
        "module": os.path.realpath(solweig_gpu.__file__), "root": root, "rev": rev,
        "data": os.path.abspath(args.data), "device": str(a.device),
        "shape": list(a.shape), "scale": float(scale),
        "amaxvalue": float(amaxvalue), "bush_max": float(bush.max()),
        "a_min": float(a.min()), "a_max": float(a.max()),
        "veg_frac": float((temp1 > 0).float().mean()),
        "flags": {k: v for k, v in os.environ.items() if k.startswith("SOLWEIG_")},
        "torch": torch.__version__, "warmup_s": [], "reps_s": [],
    }

    res = None
    for r in range(args.warmup + args.reps):
        if res is not None:
            del res
        sync()
        t0 = time.perf_counter()
        res = svf_calculator(2, amaxvalue=amaxvalue, a=a, vegdem=vegdem,
                             vegdem2=vegdem2, bush=bush, scale=scale,
                             save_rasters=False)
        sync()
        dt = time.perf_counter() - t0
        kind = "warmup" if r < args.warmup else "rep"
        meta["warmup_s" if r < args.warmup else "reps_s"].append(dt)
        print(f"[{args.version}/{args.tile}] {kind} {dt:.2f}s", flush=True)

    meta["max_mem_gb"] = max_mem_gb()

    if not args.no_dump:
        t0 = time.perf_counter()
        savez_fast(args.out, **{n: t.detach().cpu().numpy()
                                for n, t in zip(NAMES, res)})
        meta["dump_s"] = time.perf_counter() - t0

    if args.artifacts:
        os.makedirs(args.artifacts, exist_ok=True)
        svf_calculator(2, save_rasters=True, building_dsm_path=bdsm,
                       tree_path=tree, dem_path=dem, output_dir=args.artifacts,
                       number=args.tile)

    with open(args.out + ".json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print("DONE", json.dumps({k: meta[k] for k in
                              ("version", "rev", "tile", "device", "warmup_s", "reps_s",
                               "max_mem_gb")}),
          flush=True)


if __name__ == "__main__":
    main()
