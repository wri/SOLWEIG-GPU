#!/usr/bin/env python
"""Compare the on-disk artifact set (SkyViewFactor tif, svfs zip members,
shadowmats npz members) between two artifact dirs for one tile key."""
import sys
import zipfile

import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def tif_arr(path):
    ds = gdal.Open(path)
    arr = ds.GetRasterBand(1).ReadAsArray()
    gt, pr = ds.GetGeoTransform(), ds.GetProjection()
    ds = None
    return arr, gt, pr


def main():
    d_a, d_b, key = sys.argv[1], sys.argv[2], sys.argv[3]
    worst = 0.0
    a, gta, pra = tif_arr(f"{d_a}/SkyViewFactor_{key}.tif")
    b, gtb, prb = tif_arr(f"{d_b}/SkyViewFactor_{key}.tif")
    d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    worst = max(worst, d)
    print(f"SkyViewFactor tif: max|d|={d:.3e} geo_equal={gta == gtb and pra == prb}")

    import tempfile, os
    for zn in ("svfs_%s.zip" % key,):
        with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
            za, zb = zipfile.ZipFile(f"{d_a}/{zn}"), zipfile.ZipFile(f"{d_b}/{zn}")
            na, nb = sorted(za.namelist()), sorted(zb.namelist())
            print(f"{zn}: members_equal={na == nb} n={len(na)}")
            za.extractall(ta); zb.extractall(tb)
            for m in na:
                x, _, _ = tif_arr(os.path.join(ta, m))
                y, _, _ = tif_arr(os.path.join(tb, m))
                dm = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())
                worst = max(worst, dm)
                if dm > 0:
                    print(f"  {m}: max|d|={dm:.3e}")

    A = np.load(f"{d_a}/shadowmats_{key}.npz")
    B = np.load(f"{d_b}/shadowmats_{key}.npz")
    print(f"shadowmats keys: {sorted(A.files)} == {sorted(B.files)}: {sorted(A.files) == sorted(B.files)}")
    for k in A.files:
        eq = np.array_equal(A[k], B[k])
        print(f"  {k}: bit_equal={eq}")
        if not eq:
            worst = max(worst, float(np.abs(A[k] - B[k]).max()))
    print(f"ARTIFACTS worst max|d| across non-bit-equal members: {worst:.3e} "
          f"({'PASS (LSB only)' if worst <= 2e-5 else '*** FAIL ***'})")


if __name__ == "__main__":
    main()
