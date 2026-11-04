#!/usr/bin/env python
"""Run svf_calculator for one tile with one package version; dump all 19 returned
arrays (lossless) + timing/meta JSON. Version isolation: 'orig' prepends the
original repo root to sys.path, which beats the editable-install meta-path finder.
"""
import argparse
import json
import os
import sys
import time
import zipfile

ORIG_ROOT = "/home/arsh/wri/code/SOLWEIG-GPU-main_1/SOLWEIG-GPU-main"
CUR_ROOT = "/home/arsh/wri/code/SOLWEIG-GPU-main"
DEFAULT_DATA = os.path.join(CUR_ROOT, "data/test1/processed_inputs")

# Exact return order of svf_calculator (identical in both versions).
NAMES = ["svf", "svfaveg", "svfE", "svfEaveg", "svfEveg", "svfN", "svfNaveg",
         "svfNveg", "svfS", "svfSaveg", "svfSveg", "svfveg", "svfW", "svfWaveg",
         "svfWveg", "vegshmat", "vbshvegshmat", "shmat", "SVFtotal"]


def savez_fast(path, **arrays):
    """Lossless npz (valid for np.load) with fast deflate; harness-only helper."""
    import numpy as np
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for name, arr in arrays.items():
            with zf.open(name + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(arr), allow_pickle=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["orig", "cur"], required=True)
    ap.add_argument("--tile", required=True)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--out", required=True, help="output .npz path (meta at .json)")
    ap.add_argument("--no-dump", action="store_true", help="timing only")
    ap.add_argument("--artifacts", default=None,
                    help="also run save_rasters=True standalone mode into this dir")
    args = ap.parse_args()

    if args.version == "orig":
        sys.path.insert(0, ORIG_ROOT)
    import solweig_gpu
    expected = {"orig": ORIG_ROOT, "cur": CUR_ROOT}[args.version]
    actual = os.path.realpath(solweig_gpu.__file__)
    assert actual.startswith(os.path.realpath(expected) + os.sep), \
        f"import resolved to {actual}, expected under {expected}"

    import numpy as np
    import torch
    from solweig_gpu.shadow import svf_calculator, load_raster_to_tensor

    bdsm = os.path.join(args.data, "Building_DSM", f"Building_DSM_{args.tile}.tif")
    tree = os.path.join(args.data, "Trees", f"Trees_{args.tile}.tif")
    dem = os.path.join(args.data, "DEM", f"DEM_{args.tile}.tif")

    # ---- Input prep: verbatim replica of svf_calculator standalone mode
    # (shadow.py:582-603 current == :455-476 original).
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
        "version": args.version, "tile": args.tile, "module": actual,
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
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = svf_calculator(2, amaxvalue=amaxvalue, a=a, vegdem=vegdem,
                             vegdem2=vegdem2, bush=bush, scale=scale,
                             save_rasters=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        kind = "warmup" if r < args.warmup else "rep"
        meta["warmup_s" if r < args.warmup else "reps_s"].append(dt)
        print(f"[{args.version}/{args.tile}] {kind} {dt:.2f}s", flush=True)

    meta["max_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9

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
                              ("version", "tile", "warmup_s", "reps_s", "max_mem_gb")}),
          flush=True)


if __name__ == "__main__":
    main()
