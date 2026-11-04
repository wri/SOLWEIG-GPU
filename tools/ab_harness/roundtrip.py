#!/usr/bin/env python
"""Prove the SVF cache written by the NEW code loads through the real pipeline
consumer (utci_process.load_cached_svf_outputs + zip/vsizip readers) and matches
the in-memory dump bit-for-bit.
Usage: roundtrip.py <artifacts_dir> <tile_key> <dump_npz>
"""
import os
import shutil
import sys
import tempfile

import numpy as np

CUR_ROOT = "/home/arsh/wri/code/SOLWEIG-GPU-main"
DATA = os.path.join(CUR_ROOT, "data/test1/processed_inputs")


def main():
    artifacts, key, dump_npz = sys.argv[1], sys.argv[2], sys.argv[3]
    import solweig_gpu
    assert "SOLWEIG-GPU-main_1" not in solweig_gpu.__file__
    from solweig_gpu.utci_process import load_cached_svf_outputs

    with tempfile.TemporaryDirectory() as td:
        os.makedirs(f"{td}/Building_DSM")
        os.makedirs(f"{td}/SVF")
        shutil.copy(f"{DATA}/Building_DSM/Building_DSM_{key}.tif", f"{td}/Building_DSM/")
        for f in os.listdir(artifacts):
            shutil.copy(os.path.join(artifacts, f), f"{td}/SVF/")
        out = load_cached_svf_outputs(f"{td}/Building_DSM/Building_DSM_{key}.tif", key)

    # loader returns the same 19-tuple order as svf_calculator
    NAMES = ["svf", "svfaveg", "svfE", "svfEaveg", "svfEveg", "svfN", "svfNaveg",
             "svfNveg", "svfS", "svfSaveg", "svfSveg", "svfveg", "svfW",
             "svfWaveg", "svfWveg", "vegshmat", "vbshvegshmat", "shmat", "SVFtotal"]
    assert len(out) == len(NAMES), f"loader returned {len(out)} values"
    named = dict(zip(NAMES, out))

    D = np.load(dump_npz)
    checked, ok = 0, True
    for name, tensor in named.items():
        arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        dump_name = {"shmat": "shmat", "vegshmat": "vegshmat",
                     "vbshmat": "vbshvegshmat", "vbshvegshmat": "vbshvegshmat"}.get(name, name)
        if dump_name not in D.files:
            print(f"  (skip {name}: no dump counterpart)")
            continue
        eq = np.array_equal(arr, D[dump_name])
        checked += 1
        ok &= eq
        if not eq:
            d = np.abs(arr.astype(np.float64) - D[dump_name].astype(np.float64)).max()
            print(f"  *** {name}: NOT bit-equal to dump (max|d|={d:.3e})")
    print(f"ROUNDTRIP {'PASS' if ok else 'FAIL'}: {checked} arrays loaded via "
          f"load_cached_svf_outputs match the in-memory dump bit-for-bit")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
