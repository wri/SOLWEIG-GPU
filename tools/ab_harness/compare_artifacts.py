#!/usr/bin/env python
"""Compare the on-disk artifact set (SkyViewFactor tif, svfs zip members,
shadowmats npz members) between two artifact dirs for one tile key.
Usage: compare_artifacts.py <dir_a> <dir_b> <tile_key>
PASS (exit 0) = same member sets, equal geo metadata, bit-equal shadow matrices and
every raster within float32 summation-order rounding (max|d| <= 2e-5). When exactly
one side predates upstream's svfN fix (its svfN/svfNveg/svfNaveg are all zero),
those three members are reported as expected differences and left out of the verdict.
"""
import sys
import zipfile

import numpy as np
from osgeo import gdal

from _common import N_ARRAYS, n_branch_dead

gdal.UseExceptions()
TOL = 2e-5


def tif_arr(path):
    ds = gdal.Open(path)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    gt, pr = ds.GetGeoTransform(), ds.GetProjection()
    ds = None
    return arr, gt, pr


def member(zip_path, name):
    return tif_arr(f"/vsizip/{zip_path}/{name}")[0]


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    d_a, d_b, key = sys.argv[1:4]
    ok, worst = True, 0.0

    a, gta, pra = tif_arr(f"{d_a}/SkyViewFactor_{key}.tif")
    b, gtb, prb = tif_arr(f"{d_b}/SkyViewFactor_{key}.tif")
    d = float(np.abs(a - b).max())
    geo = gta == gtb and pra == prb
    ok &= geo
    worst = max(worst, d)
    print(f"SkyViewFactor tif: max|d|={d:.3e} geo_equal={geo}")

    zn = f"svfs_{key}.zip"
    za, zb = f"{d_a}/{zn}", f"{d_b}/{zn}"
    with zipfile.ZipFile(za) as z:
        na = sorted(z.namelist())
    with zipfile.ZipFile(zb) as z:
        nb = sorted(z.namelist())
    ok &= na == nb
    print(f"{zn}: members_equal={na == nb} n={len(na)}")
    dead_a = n_branch_dead(lambda n: member(za, f"{n}.tif"))
    dead_b = n_branch_dead(lambda n: member(zb, f"{n}.tif"))
    expected = {f"{n}.tif" for n in N_ARRAYS} if dead_a != dead_b else set()
    if expected:
        print(f"  NOTE: {'A' if dead_a else 'B'} predates upstream's svfN fix "
              "(svfN/svfNveg/svfNaveg all zero); those members are expected to differ")
    for m in na:
        if m not in nb:
            continue
        dm = float(np.abs(member(za, m) - member(zb, m)).max())
        if m in expected:
            print(f"  {m}: max|d|={dm:.3e} EXPECTED DIFF (svfN fix, upstream v2.0.0)")
            continue
        worst = max(worst, dm)
        if dm > 0:
            print(f"  {m}: max|d|={dm:.3e}")

    A = np.load(f"{d_a}/shadowmats_{key}.npz")
    B = np.load(f"{d_b}/shadowmats_{key}.npz")
    keys_equal = sorted(A.files) == sorted(B.files)
    ok &= keys_equal
    print(f"shadowmats keys: {sorted(A.files)} == {sorted(B.files)}: {keys_equal}")
    for k in A.files:
        if k not in B.files:
            continue
        eq = np.array_equal(A[k], B[k])
        ok &= eq  # shadow matrices must be bit-identical
        print(f"  {k}: bit_equal={eq}")
        if not eq:
            worst = max(worst, float(np.abs(A[k].astype(np.float64)
                                            - B[k].astype(np.float64)).max()))
    ok &= worst <= TOL
    print(f"ARTIFACTS worst max|d| across compared members: {worst:.3e} "
          f"({'PASS' if ok else '*** FAIL ***'})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
