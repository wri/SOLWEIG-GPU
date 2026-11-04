#!/usr/bin/env python
"""Prove the svf* diffs between orig and cur are pure float32 rounding.

Both versions produced BIT-IDENTICAL shadow matrices; they differ only in the
order of the weighted float32 accumulation (orig: 12 ring-adds per patch;
cur: precomputed band weight-sum). This script rebuilds every svf* array in
FLOAT64 from the shared shadow matrices and the exact patch/weight definitions,
then measures how far each version's float32 result is from the float64 truth.
If both are within a few float32 ulps, the difference is rounding, full stop.

Usage: validate_lsb.py <orig.npz> <cur.npz> <tile_key> [data_dir]
"""
import sys

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

DATA_DEFAULT = "/home/arsh/wri/code/SOLWEIG-GPU-main/data/test1/processed_inputs"

# patch_option=2 (identical in both versions)
ANNULINO = [0, 12, 24, 36, 48, 60, 72, 84, 90]
ALTS = [6, 18, 30, 42, 54, 66, 78, 90]
PATCHES = [31, 30, 28, 24, 19, 13, 7, 1]
AZISTART = [0, 4, 2, 5, 8, 0, 10, 0]


def annulus_weight64(k, aziinterval):
    steprad = (360.0 / aziinterval) * (np.pi / 180.0)
    annulus = 91.0 - k
    w = (1.0 / (2.0 * np.pi)) * np.sin(np.pi / 180.0) * \
        np.sin((np.pi * (2.0 * annulus - 1.0)) / 180.0)
    return steprad * w


def main():
    orig_npz, cur_npz, tile = sys.argv[1], sys.argv[2], sys.argv[3]
    data = sys.argv[4] if len(sys.argv) > 4 else DATA_DEFAULT
    O, C = np.load(orig_npz), np.load(cur_npz)

    for m in ("shmat", "vegshmat", "vbshvegshmat"):
        assert np.array_equal(O[m], C[m]), f"{m} not bit-equal; rounding proof invalid"
    shmat = O["shmat"].astype(np.float64)
    vegshmat = O["vegshmat"].astype(np.float64)
    vbshmat = O["vbshvegshmat"].astype(np.float64)
    rows, cols = shmat.shape[:2]

    # vegdem2 for the `last` term, replicated from the shared prep (float32 ops)
    def rd(p):
        ds = gdal.Open(p)
        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        return arr

    a = rd(f"{data}/Building_DSM/Building_DSM_{tile}.tif")
    temp1 = rd(f"{data}/Trees/Trees_{tile}.tif")
    temp1[temp1 < 0.0] = 0.0
    vegdem2 = (temp1 * np.float32(0.25) + a).astype(np.float32)
    vegdem2[vegdem2 == a] = 0.0

    # azimuth per patch: replicate the code's ACTUAL iazimuth table, including its
    # float32 quirk — `int(360 / (360/patches)_f32)` truncates bands 1,2,5,6 to
    # 29/27/12/6 entries, so the table misaligns vs nominal bands and the last 4
    # slots stay 0.0. Identical in both versions; conditions must see these values.
    import torch
    aziint_t = torch.tensor(PATCHES)
    azint_f32 = torch.tensor([360 / p for p in aziint_t])
    iaz_t = torch.zeros(sum(PATCHES))
    idx = 0
    for j in range(8):
        for k in range(int(360 / azint_f32[j])):
            v = k * azint_f32[j] + AZISTART[j]
            if v > 360.0:
                v = v - 360.0
            iaz_t[idx] = v
            idx += 1
    iaz = [float(v) for v in iaz_t]

    z = lambda: np.zeros((rows, cols), dtype=np.float64)
    svf, svfE, svfS, svfW, svfN = z(), z(), z(), z(), z()
    svfveg, svfEveg, svfSveg, svfWveg, svfNveg = z(), z(), z(), z(), z()
    svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg = z(), z(), z(), z(), z()

    index = 0
    for i in range(8):
        wf = sum(annulus_weight64(k, PATCHES[i])
                 for k in range(ANNULINO[i] + 1, ANNULINO[i + 1] + 1))
        wa = sum(annulus_weight64(k, np.ceil(PATCHES[i] / 2.0))
                 for k in range(ANNULINO[i] + 1, ANNULINO[i + 1] + 1))
        for j in range(PATCHES[i]):
            azimuth = iaz[index]
            sh = shmat[:, :, index]
            veg = vegshmat[:, :, index]
            vbs = vbshmat[:, :, index]
            svf += wf * sh
            svfveg += wf * veg
            svfaveg += wf * vbs
            if 0 <= azimuth < 180:
                svfE += wa * sh
                svfEveg += wa * veg
                svfEaveg += wa * vbs
            if 90 <= azimuth < 270:
                svfS += wa * sh
                svfSveg += wa * veg
                svfSaveg += wa * vbs
            if 180 <= azimuth < 360:
                svfW += wa * sh
                svfWveg += wa * veg
                svfWaveg += wa * vbs
            # N branch is dead in both versions (270 <= az < 90 never true)
            index += 1

    svfS += 3.0459e-4
    svfW += 3.0459e-4
    last = np.where(vegdem2 == 0.0, 3.0459e-4, 0.0)
    svfSveg += last
    svfWveg += last
    svfSaveg += last
    svfWaveg += last
    ref = {"svf": svf, "svfE": svfE, "svfS": svfS, "svfW": svfW, "svfN": svfN,
           "svfveg": svfveg, "svfEveg": svfEveg, "svfSveg": svfSveg,
           "svfWveg": svfWveg, "svfNveg": svfNveg, "svfaveg": svfaveg,
           "svfEaveg": svfEaveg, "svfSaveg": svfSaveg, "svfWaveg": svfWaveg,
           "svfNaveg": svfNaveg}
    for k in ref:
        ref[k] = np.minimum(ref[k], 1.0)
    ref["SVFtotal"] = ref["svf"] - (1 - ref["svfveg"]) * (1 - 0.03)

    print(f"{'array':12s} {'|orig-ref|max':>14s} {'|cur-ref|max':>14s} "
          f"{'|orig-cur|max':>14s}  (float32 eps at 1.0 = 1.19e-07)")
    worst = 0.0
    for n, r in ref.items():
        do = float(np.abs(O[n].astype(np.float64) - r).max())
        dc = float(np.abs(C[n].astype(np.float64) - r).max())
        db = float(np.abs(O[n].astype(np.float64) - C[n].astype(np.float64)).max())
        worst = max(worst, do, dc)
        print(f"{n:12s} {do:14.3e} {dc:14.3e} {db:14.3e}")
    print(f"\nWorst distance from float64 truth (either version): {worst:.3e} "
          f"= {worst/1.19e-07:.1f} float32 ulps at 1.0")
    print("CONCLUSION:", "both versions are correct float32 roundings of the same "
          "mathematical result — differences are summation-order rounding only"
          if worst < 5e-6 else "*** diffs exceed rounding scale — investigate ***")


if __name__ == "__main__":
    main()
