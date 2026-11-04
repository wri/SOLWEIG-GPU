#!/usr/bin/env python
"""Compare two dump_svf.py npz dumps (A=reference/orig, B=candidate/cur).
Usage: compare_svf.py <a.npz> <b.npz> [label] [strict]
Bit-exact expected: vegshmat, vbshvegshmat, shmat.
Tiny LSB diffs allowed on the 15 svf* arrays + SVFtotal (float32 reassociation
from the per-band weight-sum precompute): PASS if max|d| <= 2e-5, mean|d| <= 5e-6.
When exactly one dump predates upstream's svfN fix (its svfN/svfNveg/svfNaveg are
all zero), those three arrays are reported as expected differences and left out of
the verdict, except in strict mode (a flag toggle must never change them).
"""
import json
import sys

import numpy as np

from _common import N_ARRAYS, n_branch_dead

MATS = {"vegshmat", "vbshvegshmat", "shmat"}
# Tolerances for the svf* accumulations: validate_lsb.py proved (via a float64
# reference rebuilt from the bit-identical shadow mats) that both versions are
# correct float32 roundings of the same sum — orig within 3.9e-6 of truth, cur
# within 7.7e-7. Allow up to ~2x the observed orig-vs-truth distance.
MAXTOL, MEANTOL = 2e-5, 5e-6


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    strict = len(sys.argv) > 4 and sys.argv[4] == "strict"  # all arrays must be bit-equal
    A, B = np.load(a_path), np.load(b_path)
    assert set(A.files) == set(B.files), (A.files, B.files)
    dead_a, dead_b = n_branch_dead(A.__getitem__), n_branch_dead(B.__getitem__)
    expected = set(N_ARRAYS) if dead_a != dead_b and not strict else set()
    report, ok = {}, True
    for n in sorted(A.files):
        a, b = A[n], B[n]
        assert a.shape == b.shape and a.dtype == b.dtype, (n, a.shape, b.shape, a.dtype, b.dtype)
        nan_a, nan_b = int(np.isnan(a).sum()), int(np.isnan(b).sum())
        if np.array_equal(a, b):
            report[n] = {"bit_equal": True, "nan": [nan_a, nan_b]}
            continue
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        neq = a != b
        idx = np.unravel_index(int(np.argmax(d)), d.shape)
        r = {"bit_equal": False, "nan": [nan_a, nan_b],
             "n_diff": int(neq.sum()), "n_total": int(a.size),
             "pct_diff": round(100.0 * neq.sum() / a.size, 4),
             "max_abs": float(d.max()), "mean_abs": float(d.mean()),
             "mean_abs_diffpx": float(d[neq].mean()),
             "argmax": [int(i) for i in idx],
             "vals_at_argmax": [float(a[idx]), float(b[idx])]}
        if n in expected:
            r["expected_diff"] = True  # svfN fix; not part of the verdict
            report[n] = r
            continue
        passed = (not strict) and (n not in MATS) and r["max_abs"] <= MAXTOL and r["mean_abs"] <= MEANTOL
        r["pass"] = bool(passed)
        ok &= passed
        report[n] = r

    n_biteq = sum(1 for r in report.values() if r.get("bit_equal"))
    print(f"=== compare {label}: {a_path} vs {b_path}")
    if expected:
        print(f"  NOTE: {'A' if dead_a else 'B'} predates upstream's svfN fix "
              "(svfN/svfNveg/svfNaveg all zero); those arrays are expected to differ")
    for n, r in report.items():
        if r.get("bit_equal"):
            print(f"  {n:14s} BIT-EQUAL")
        else:
            status = ("EXPECTED DIFF (svfN fix, upstream v2.0.0)" if r.get("expected_diff")
                      else "PASS(lsb)" if r["pass"] else "*** FAIL ***")
            print(f"  {n:14s} DIFF n={r['n_diff']}/{r['n_total']} ({r['pct_diff']}%) "
                  f"max|d|={r['max_abs']:.3e} mean|d|={r['mean_abs']:.3e} {status}")
    verdict = "PASS" if ok else "FAIL"
    print(f"RESULT {label}: {verdict} ({n_biteq}/{len(report)} arrays bit-equal)")
    out = a_path.replace(".npz", f"") + f"__vs__{b_path.split('/')[-1].replace('.npz','')}.cmp.json"
    with open(out, "w") as fh:
        json.dump({"a": a_path, "b": b_path, "pass": ok, "expected_diff": sorted(expected),
                   "report": report}, fh, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
