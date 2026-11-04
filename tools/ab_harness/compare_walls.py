#!/usr/bin/env python
"""Compare walls_/aspect_ GeoTIFF pairs produced by the two versions.
Usage: compare_walls.py <orig_out_dir> <cur_out_dir>
Expects <dir>/walls/walls_<key>.tif and <dir>/aspect/aspect_<key>.tif.
PASS = every pair bit-identical (array + geotransform + projection).
Any walls diff is tested against the walllimit=3.0 float64-vs-float32 tie signature.
"""
import json
import os
import sys

import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def read(path):
    ds = gdal.Open(path)
    arr = ds.GetRasterBand(1).ReadAsArray()
    gt, proj = ds.GetGeoTransform(), ds.GetProjection()
    ds = None
    return arr, gt, proj


def main():
    d_a, d_b = sys.argv[1], sys.argv[2]
    ok = True
    report = {}
    for sub, prefix in (("walls", "walls_"), ("aspect", "aspect_")):
        files = sorted(f for f in os.listdir(os.path.join(d_a, sub)) if f.endswith(".tif"))
        files_b = sorted(f for f in os.listdir(os.path.join(d_b, sub)) if f.endswith(".tif"))
        if files != files_b:
            print(f"*** file set mismatch in {sub}: {set(files) ^ set(files_b)}")
            ok = False
        for f in files:
            a, gt_a, pr_a = read(os.path.join(d_a, sub, f))
            b, gt_b, pr_b = read(os.path.join(d_b, sub, f))
            key = f"{sub}/{f}"
            geo_ok = (gt_a == gt_b) and (pr_a == pr_b)
            if np.array_equal(a, b) and geo_ok:
                report[key] = {"bit_equal": True}
                continue
            ok = False
            neq = a != b
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            r = {"bit_equal": False, "geo_equal": geo_ok,
                 "n_diff": int(neq.sum()), "n_total": int(a.size),
                 "max_abs": float(d.max())}
            if sub == "walls" and neq.any():
                ys, xs = np.where(neq)
                r["sample_pairs"] = [[float(a[y, x]), float(b[y, x])]
                                     for y, x in list(zip(ys, xs))[:5]]
            report[key] = r
            print(f"  *** {key}: n_diff={r['n_diff']} max|d|={r['max_abs']:.3e}")

    n_eq = sum(1 for r in report.values() if r.get("bit_equal"))
    print(f"RESULT walls+aspect: {'PASS' if ok else 'FAIL'} ({n_eq}/{len(report)} files bit-equal)")
    with open(os.path.join(d_b, "compare_walls.json"), "w") as fh:
        json.dump({"pass": ok, "report": report}, fh, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
