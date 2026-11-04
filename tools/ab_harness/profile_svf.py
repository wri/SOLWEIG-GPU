#!/usr/bin/env python
"""Profile CURRENT svf_calculator: per-region wall times (march fill vs rest vs
save), per-band shadow_batch times, kernel counts per march step.
Run only against the CURRENT version (no --version switch).
"""
import argparse
import json
import os
import sys
import time

CUR_ROOT = "/home/arsh/wri/code/SOLWEIG-GPU-main"
DEFAULT_DATA = os.path.join(CUR_ROOT, "data/test1/processed_inputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", required=True)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", required=True, help="json report path")
    ap.add_argument("--save-dir", default=None, help="time save_svf_zip_npz_outputs into here")
    ap.add_argument("--profile-kernels", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch
    import solweig_gpu
    assert "SOLWEIG-GPU-main_1" not in solweig_gpu.__file__
    import solweig_gpu.shadow as sh
    from solweig_gpu.shadow import load_raster_to_tensor

    bdsm = os.path.join(args.data, "Building_DSM", f"Building_DSM_{args.tile}.tif")
    tree = os.path.join(args.data, "Trees", f"Trees_{args.tile}.tif")
    dem = os.path.join(args.data, "DEM", f"DEM_{args.tile}.tif")

    a, gdal_dsm = load_raster_to_tensor(bdsm)
    temp1, _ = load_raster_to_tensor(tree)
    temp2, _ = load_raster_to_tensor(dem)
    scale = 1.0 / gdal_dsm.GetGeoTransform()[1]
    temp1[temp1 < 0.0] = 0.0
    vegdem_height = temp1 + temp2
    vegdem2_height = torch.add(temp1 * 0.25, temp2)
    bush = torch.logical_not(vegdem2_height * vegdem_height) * vegdem_height
    vegdem = temp1 + a
    vegdem[vegdem == a] = 0.0
    vegdem2 = temp1 * 0.25 + a
    vegdem2[vegdem2 == a] = 0.0
    amaxvalue = torch.maximum(a.max(), vegdem_height.max())

    report = {"tile": args.tile, "shape": list(a.shape), "scale": float(scale),
              "flags": {k: v for k, v in os.environ.items() if k.startswith("SOLWEIG_")}}

    # --- wrap shadow_batch with a sync-timing decorator (batch boundaries only)
    orig_shadow_batch = sh.shadow_batch
    batch_calls = []

    def timed_shadow_batch(*a_, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_shadow_batch(*a_, **kw)
        torch.cuda.synchronize()
        batch_calls.append({"P": int(a_[5].shape[0]) if len(a_) > 5 else -1,
                            "s": time.perf_counter() - t0})
        return out

    # positional: shadow_batch(march_ceiling, a, vegdem, vegdem2, azimuths, altitude, scale, pad_to=None)
    def timed_shadow_batch_fix(march_ceiling, a_, vegdem_, vegdem2_, azimuths, altitude, scale_, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_shadow_batch(march_ceiling, a_, vegdem_, vegdem2_, azimuths, altitude, scale_, **kw)
        torch.cuda.synchronize()
        batch_calls.append({"P": int(azimuths.shape[0]), "alt": float(altitude),
                            "pad": kw.get("pad_to"), "s": round(time.perf_counter() - t0, 4)})
        return out

    sh.shadow_batch = timed_shadow_batch_fix
    try:
        # warmup (torch.compile of _shadow_core etc.)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        res = sh.svf_calculator(2, amaxvalue=amaxvalue, a=a, vegdem=vegdem,
                                vegdem2=vegdem2, bush=bush, scale=scale, save_rasters=False)
        torch.cuda.synchronize()
        report["warmup_total_s"] = time.perf_counter() - t0
        report["warmup_batch_calls"] = list(batch_calls)
        del res
        batch_calls.clear()

        torch.cuda.synchronize(); t0 = time.perf_counter()
        res = sh.svf_calculator(2, amaxvalue=amaxvalue, a=a, vegdem=vegdem,
                                vegdem2=vegdem2, bush=bush, scale=scale, save_rasters=False)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
    finally:
        sh.shadow_batch = orig_shadow_batch

    march = sum(c["s"] for c in batch_calls)
    report["total_s"] = total
    report["march_s"] = march
    report["non_march_s"] = total - march   # accumulation + weights + allocs
    report["batch_calls"] = batch_calls
    report["max_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        (svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg, svfNveg, svfS,
         svfSaveg, svfSveg, svfveg, svfW, svfWaveg, svfWveg, vegshmat,
         vbshvegshmat, shmat, SVFtotal) = res
        t0 = time.perf_counter()
        sh.save_svf_zip_npz_outputs(
            output_dir=args.save_dir, gdal_dsm=gdal_dsm, svf=svf, svfE=svfE,
            svfS=svfS, svfW=svfW, svfN=svfN, svfveg=svfveg, svfEveg=svfEveg,
            svfSveg=svfSveg, svfWveg=svfWveg, svfNveg=svfNveg, svfaveg=svfaveg,
            svfEaveg=svfEaveg, svfSaveg=svfSaveg, svfWaveg=svfWaveg,
            svfNaveg=svfNaveg, shmat=shmat, vegshmat=vegshmat,
            vbshvegshmat=vbshvegshmat, svftotal=SVFtotal, number=args.tile)
        report["save_s"] = time.perf_counter() - t0

    if args.profile_kernels:
        # short band: altitude 30 deg (band 2), 16 azimuths -> ~45 steps
        from solweig_gpu.shadow import create_patches
        _, _, annulino, skyvaultaltint, aziint, _, azistart = create_patches(2)
        hmin = a.min()
        ceiling = torch.maximum(amaxvalue, vegdem.max()) - hmin \
            if os.environ.get("SOLWEIG_MARCH_CEILING", "1") == "1" else amaxvalue
        az = torch.arange(16, device=a.device, dtype=torch.float32) * (360.0 / 28) + 2
        orig_shadow_batch(ceiling, a, vegdem, vegdem2, az, skyvaultaltint[2], scale)  # warm
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            orig_shadow_batch(ceiling, a, vegdem, vegdem2, az, skyvaultaltint[2], scale)
        torch.cuda.synchronize()
        evs = prof.key_averages()
        kernels = [(e.key, e.count, e.device_time_total) for e in evs if e.device_time_total > 0]
        n_launch = sum(c for _, c, _ in kernels)
        gpu_us = sum(t for _, _, t in kernels)
        # count steps for this batch from tanaltbyscale math
        report["kprof"] = {
            "kernel_launches": n_launch, "gpu_time_us": gpu_us,
            "top": sorted(kernels, key=lambda x: -x[2])[:12],
        }
        prof.export_chrome_trace(args.out + ".trace.json")

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("PROFILE", json.dumps({k: report[k] for k in
          ("total_s", "march_s", "non_march_s", "max_mem_gb") if k in report},
          default=float), flush=True)
    if "save_s" in report:
        print(f"save_s {report['save_s']:.2f}", flush=True)
    print("band batch times:", [c["s"] for c in batch_calls], flush=True)


if __name__ == "__main__":
    main()
