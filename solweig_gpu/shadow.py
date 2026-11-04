#SOLWEIG-GPU: GPU-accelerated SOLWEIG model for urban thermal comfort simulation
#Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np
from math import radians
from copy import deepcopy
from osgeo import gdal, osr
import datetime
import calendar
import scipy.ndimage.interpolation as sc
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate
import time
import os
import zipfile

gdal.UseExceptions()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_raster_to_tensor(dem_path, device=device):
    dataset = gdal.Open(dem_path)
    if dataset is None:
        raise FileNotFoundError(f"Could not open raster: {dem_path}")

    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray().astype(np.float32)

    return torch.tensor(array, device=device), dataset

def ensure_tensor(x, device=None):
    """
    Convert input to PyTorch tensor on specified device.
    
    Args:
        x: Input data (can be numpy array, list, or torch tensor)
        device (torch.device, optional): Target device. Auto-detects GPU if available.
    
    Returns:
        torch.Tensor: Input converted to tensor on specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device=device)
    return x

def tensor_to_numpy(x):
    """
    Move GPU/CPU torch tensor to CPU NumPy array before writing.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_raster_like_gdal(gdal_template, output_path, array):
    """
    Save a 2D array as GeoTIFF using geotransform/projection
    from the input DSM raster.
    """
    array = tensor_to_numpy(array).astype(np.float32)

    driver = gdal.GetDriverByName("GTiff")
    rows, cols = array.shape

    out_ds = driver.Create(
        output_path,
        cols,
        rows,
        1,
        gdal.GDT_Float32,
    )

    out_ds.SetGeoTransform(gdal_template.GetGeoTransform())
    out_ds.SetProjection(gdal_template.GetProjection())

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.FlushCache()

    out_ds.FlushCache()
    out_ds = None


def save_svf_zip_npz_outputs(output_dir, gdal_dsm, svf, svfE, svfS, svfW, svfN, svfveg, svfEveg, svfSveg, svfWveg, svfNveg, svfaveg, svfEaveg, svfSaveg,
    svfWaveg, svfNaveg, shmat, vegshmat, vbshvegshmat, svftotal, number=None,):
    """
    - svfs.zip containing SVF GeoTIFFs
    - shadowmats.npz containing shadow matrices
    - SkyViewFactor.tif or SkyViewFactor_<number>.tif
    """

    if output_dir is None:
        raise ValueError("output_dir must be provided when save_rasters=True")

    if gdal_dsm is None:
        raise ValueError("gdal_dsm must be provided when save_rasters=True")

    os.makedirs(output_dir, exist_ok=True)

    suffix = f"_{number}" if number is not None else ""

    zip_path = os.path.join(output_dir, f"svfs{suffix}.zip")
    npz_path = os.path.join(output_dir, f"shadowmats{suffix}.npz")
    svftotal_path = os.path.join(output_dir, f"SkyViewFactor{suffix}.tif")

    if os.path.isfile(zip_path):
        os.remove(zip_path)

    rasters_to_save = {
        "svf.tif": svf,
        "svfE.tif": svfE,
        "svfS.tif": svfS,
        "svfW.tif": svfW,
        "svfN.tif": svfN,
        "svfveg.tif": svfveg,
        "svfEveg.tif": svfEveg,
        "svfSveg.tif": svfSveg,
        "svfWveg.tif": svfWveg,
        "svfNveg.tif": svfNveg,
        "svfaveg.tif": svfaveg,
        "svfEaveg.tif": svfEaveg,
        "svfSaveg.tif": svfSaveg,
        "svfWaveg.tif": svfWaveg,
        "svfNaveg.tif": svfNaveg,
    }

    temporary_tifs = []

    for tif_name, data in rasters_to_save.items():
        tif_path = os.path.join(output_dir, tif_name)
        save_raster_like_gdal(gdal_dsm, tif_path, data)
        temporary_tifs.append(tif_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for tif_path in temporary_tifs:
            zf.write(tif_path, os.path.basename(tif_path))

    for tif_path in temporary_tifs:
        os.remove(tif_path)

    save_raster_like_gdal(gdal_dsm, svftotal_path, svftotal)

    mats = {
        "shadowmat": np.asarray(tensor_to_numpy(shmat), dtype=np.float32),
        "vegshadowmat": np.asarray(tensor_to_numpy(vegshmat), dtype=np.float32),
        "vbshmat": np.asarray(tensor_to_numpy(vbshvegshmat), dtype=np.float32),
    }
    # Same arrays, same npz keys, still a np.load-compatible npz — only the
    # archive's deflate level changes. np.savez_compressed hardwires zlib level 6,
    # which costs 10-25 s/tile on the ~2 GB 0/1 shadow-mat trio; level 1
    # compresses these masks nearly as well at a fraction of the CPU cost.
    # Set SOLWEIG_NPZ_COMPRESSLEVEL=default to restore np.savez_compressed.
    _level = os.environ.get("SOLWEIG_NPZ_COMPRESSLEVEL", "1")
    if _level == "default":
        np.savez_compressed(npz_path, **mats)
    else:
        with zipfile.ZipFile(npz_path, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=int(_level)) as zf:
            for _name, _arr in mats.items():
                with zf.open(_name + ".npy", "w", force_zip64=True) as _f:
                    np.lib.format.write_array(_f, _arr, allow_pickle=False)

def _shadow_core(f, temp, a, tempvegdem, tempvegdem2, vegsh, vbshvegsh):
    """
    Per-march-step elementwise core (operates on fixed full-tile shapes).
    Extracted so torch.compile can fuse these ~8 memory-bound elementwise ops into
    1-2 kernels; the dynamic int()/slice shifting stays in the eager caller.
    Functionally identical to the original in-place updates.
    """
    f = torch.max(f, temp)
    sh = (f > a).float()                              # sh[f>a]=1.; sh[f<=a]=0.
    fabovea = tempvegdem > a
    gabovea = tempvegdem2 > a
    vegsh2 = fabovea.float() - gabovea.float()
    vegsh = torch.max(vegsh, vegsh2)
    vegsh = vegsh.masked_fill(vegsh * sh > 0., 0.0)   # vegsh[(vegsh*sh>0.)]=0.
    vbshvegsh = vegsh + vbshvegsh
    return f, sh, fabovea, vegsh, vbshvegsh


# Fixed shapes per tile -> compiles once and is replayed for every march step.
# Set SOLWEIG_COMPILE_MARCH=0 to force the eager path (for A/B benchmarking).
if torch.cuda.is_available() and os.environ.get("SOLWEIG_COMPILE_MARCH", "1") == "1":
    try:
        import torch._dynamo
        # Unattended run: if some tile shape ever fails to compile, fall back to
        # eager for that shape instead of crashing the whole job.
        torch._dynamo.config.suppress_errors = True
        _shadow_core_dispatch = torch.compile(_shadow_core, dynamic=False)
    except Exception as _e:  # pragma: no cover
        print(f"[WARN] torch.compile unavailable ({_e}); using eager march core.")
        _shadow_core_dispatch = _shadow_core
else:
    _shadow_core_dispatch = _shadow_core


def _batch_step_fast(f, vegsh, vbshvegsh, a, vegdem, vegdem2, rows, cols, dxi, dyi, dz, m):
    """
    One batched march step (steps >= 2), fast-freeze variant (requires a.min()>=0):
    per-lane shift indices + bounds mask + 3 shifted gathers + _shadow_core, with
    only the vbshvegsh accumulator masked by the per-lane active mask `m`.
    Bit-identical op sequence to shadow_batch's eager while-loop body; kept
    tensor-only (no Python scalars that vary per step) so torch.compile records
    ONE graph per (P,H,W) shape and replays it for every step.
    """
    H, W = a.shape[0], a.shape[1]
    src_r = rows[None, :] + dxi[:, None]                     # (P,H)
    src_c = cols[None, :] + dyi[:, None]                     # (P,W)
    valid = ((src_r >= 0) & (src_r < H))[:, :, None] & ((src_c >= 0) & (src_c < W))[:, None, :]
    idx_r = src_r.clamp(0, H - 1)[:, :, None]                # (P,H,1)
    idx_c = src_c.clamp(0, W - 1)[:, None, :]                # (P,1,W)
    dz3 = dz[:, None, None]
    zero = torch.zeros((), device=a.device)
    temp = torch.where(valid, a[idx_r, idx_c] - dz3, zero)
    tempvegdem = torch.where(valid, vegdem[idx_r, idx_c] - dz3, zero)
    tempvegdem2 = torch.where(valid, vegdem2[idx_r, idx_c] - dz3, zero)
    nf, nsh, _fab, nvegsh, nvbsh = _shadow_core(f, temp, a, tempvegdem, tempvegdem2, vegsh, vbshvegsh)
    nvbsh = torch.where(m[:, None, None], nvbsh, vbshvegsh)
    return nf, nsh, nvegsh, nvbsh


def _batch_step_full(f, sh, vegsh, vbshvegsh, a, vegdem, vegdem2, rows, cols, dxi, dyi, dz, m):
    """4-way-freeze variant of _batch_step_fast (used when a.min() < 0)."""
    H, W = a.shape[0], a.shape[1]
    src_r = rows[None, :] + dxi[:, None]
    src_c = cols[None, :] + dyi[:, None]
    valid = ((src_r >= 0) & (src_r < H))[:, :, None] & ((src_c >= 0) & (src_c < W))[:, None, :]
    idx_r = src_r.clamp(0, H - 1)[:, :, None]
    idx_c = src_c.clamp(0, W - 1)[:, None, :]
    dz3 = dz[:, None, None]
    zero = torch.zeros((), device=a.device)
    temp = torch.where(valid, a[idx_r, idx_c] - dz3, zero)
    tempvegdem = torch.where(valid, vegdem[idx_r, idx_c] - dz3, zero)
    tempvegdem2 = torch.where(valid, vegdem2[idx_r, idx_c] - dz3, zero)
    nf, nsh, _fab, nvegsh, nvbsh = _shadow_core(f, temp, a, tempvegdem, tempvegdem2, vegsh, vbshvegsh)
    m3 = m[:, None, None]
    return (torch.where(m3, nf, f), torch.where(m3, nsh, sh),
            torch.where(m3, nvegsh, vegsh), torch.where(m3, nvbsh, vbshvegsh))


# Compile the whole batched step as ONE graph per (P,H,W) shape: the gathers fuse
# with the subtract/where and the core, collapsing ~180 eager kernel launches per
# step into a handful. Every op is elementwise/gather/compare (no a*b+c pattern),
# so inductor cannot FMA-contract anything: bit-identical to eager. Set
# SOLWEIG_COMPILE_BATCH=0 to force the eager step (A/B benchmarking).
if torch.cuda.is_available() and os.environ.get("SOLWEIG_COMPILE_BATCH", "1") == "1":
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        # Unpadded tail batches can produce ~9 distinct P shapes per tile size;
        # keep headroom so shape churn never evicts hot graphs.
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        _batch_step_fast_dispatch = torch.compile(_batch_step_fast, dynamic=False)
        _batch_step_full_dispatch = torch.compile(_batch_step_full, dynamic=False)
    except Exception as _e:  # pragma: no cover
        print(f"[WARN] torch.compile unavailable ({_e}); using eager batch step.")
        _batch_step_fast_dispatch = _batch_step_fast
        _batch_step_full_dispatch = _batch_step_full
else:
    _batch_step_fast_dispatch = _batch_step_fast
    _batch_step_full_dispatch = _batch_step_full


def shadow(amaxvalue, a, vegdem, vegdem2, bush, azimuth, altitude, scale):
    """
    Calculate shadow patterns from buildings and vegetation using GPU-accelerated ray tracing.
    
    This function performs GPU-accelerated shadow calculations by tracing sun rays
    across the Digital Surface Model (DSM) accounting for buildings and vegetation.
    
    Args:
        amaxvalue (torch.Tensor): Maximum elevation value in the domain
        a (torch.Tensor): Digital Surface Model (DSM) array
        vegdem (torch.Tensor): Vegetation canopy DSM
        vegdem2 (torch.Tensor): Vegetation trunk zone DSM
        bush (torch.Tensor): Bush/shrub layer DSM
        azimuth (float): Solar azimuth angle (degrees, 0=North, clockwise)
        altitude (float): Solar altitude angle (degrees above horizon)
        scale (float): Grid resolution in pixels per meter
    
    Returns:
        tuple: (sh, vegsh, vbshvegsh) where:
            - sh: Shadow map (0=shadow, 1=sunlit)
            - vegsh: Vegetation shadow influence
            - vbshvegsh: Combined vegetation and building shadow
    
    Notes:
        - Automatically uses GPU if available, otherwise CPU
        - Implements anisotropic shadow casting
        - Accounts for vegetation transmittance
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    degrees = torch.pi / 180.
    if azimuth == 0.0:
        azimuth = 1e-12
    azimuth = ensure_tensor(azimuth)
    altitude = ensure_tensor(altitude)
    azimuth = azimuth * degrees #torch.tensor(azimuth * degrees, device=a.device)
    altitude = altitude * degrees #torch.tensor(altitude * degrees, device=a.device)

    dx = 0.
    dy = 0.
    dz = 0.
    sizex = a.shape[0]
    sizey = a.shape[1]

    device = a.device

    dx = torch.tensor(dx, device=device)
    dy = torch.tensor(dy, device=device)
    dz = torch.tensor(dz, device=device)

    temp = torch.zeros((sizex, sizey), device=device)
    tempvegdem = torch.zeros((sizex, sizey), device=device)
    tempvegdem2 = torch.zeros((sizex, sizey), device=device)
    sh = torch.zeros((sizex, sizey), device=device)
    vbshvegsh = torch.zeros((sizex, sizey), device=device)
    tempbush = torch.zeros((sizex, sizey), device=device)

    f = a.clone()
    g = torch.zeros((sizex, sizey), device=device)
    bushplant = (bush > 1.).float()
    # Evaluate once as a Python bool (single host sync) so the per-march-step
    # guards below short-circuit without a GPU->CPU sync on every iteration.
    bush_present = bool(bush.max() > 0.)
    vegsh = torch.zeros((sizex, sizey), device=device) + bushplant

    pibyfour = torch.pi / 4.
    threetimespibyfour = 3. * pibyfour
    fivetimespibyfour = 5. * pibyfour
    seventimespibyfour = 7. * pibyfour
    sinazimuth = torch.sin(azimuth)
    cosazimuth = torch.cos(azimuth)
    tanazimuth = torch.tan(azimuth)
    signsinazimuth = torch.sign(sinazimuth)
    signcosazimuth = torch.sign(cosazimuth)
    dssin = torch.abs((1. / sinazimuth))
    dscos = torch.abs((1. / cosazimuth))
    tanaltitudebyscale = torch.tan(altitude) / scale

    index = 1
    fabovea = None

    while (amaxvalue >= dz and torch.abs(dx) < sizex and torch.abs(dy) < sizey):
        if (pibyfour <= azimuth < threetimespibyfour or fivetimespibyfour <= azimuth < seventimespibyfour):
            dy = signsinazimuth * index
            dx = -1. * signcosazimuth * torch.abs(torch.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * torch.abs(torch.round(index * tanazimuth))
            dx = -1. * signcosazimuth * index
            ds = dscos

        dz = ds * index * tanaltitudebyscale

        tempvegdem.zero_()
        tempvegdem2.zero_()
        temp.zero_()
        absdx = torch.abs(dx)
        absdy = torch.abs(dy)
        xc1 = int((dx + absdx) / 2.)
        xc2 = int(sizex + (dx - absdx) / 2.)
        yc1 = int((dy + absdy) / 2.)
        yc2 = int(sizey + (dy - absdy) / 2.)
        xp1 = int(-((dx - absdx) / 2.))
        xp2 = int(sizex - (dx + absdx) / 2.)
        yp1 = int(-((dy - absdy) / 2.))
        yp2 = int(sizey - (dy + absdy) / 2.)

        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz

        f, sh, fabovea, vegsh, vbshvegsh = _shadow_core_dispatch(
            f, temp, a, tempvegdem, tempvegdem2, vegsh, vbshvegsh
        )

        if index == 1.:
            firstvegdem = tempvegdem - temp
            firstvegdem[firstvegdem <= 0.] = 1000.
            vegsh[firstvegdem < dz] = 1.
            vegsh = vegsh * (vegdem2 > a).float()
            vbshvegsh.zero_()

        if bush_present and torch.max(fabovea * bush) > 0.:
            tempbush.zero_()
            tempbush[xp1:xp2, yp1:yp2] = bush[xc1:xc2, yc1:yc2] - dz
            g = torch.max(g, tempbush)
            g *= bushplant

        index += 1.

    sh = 1. - sh
    vbshvegsh[vbshvegsh > 0.] = 1.
    vbshvegsh = vbshvegsh - vegsh

    if bush_present:
        g = g - bush
        g[g > 0.] = 1.
        g[g < 0.] = 0.
        vegsh = vegsh - bushplant + g
        vegsh[vegsh < 0.] = 0.

    vegsh[vegsh > 0.] = 1.
    vegsh = 1. - vegsh
    vbshvegsh = 1. - vbshvegsh

    del tempvegdem, tempvegdem2, temp, tempbush, fabovea
    return sh, vegsh, vbshvegsh


def shadow_batch(amaxvalue, a, vegdem, vegdem2, azimuths, altitude, scale, pad_to=None):
    """
    Batched equivalent of shadow() for P sky patches that share one altitude
    (different azimuths). Returns (sh, vegsh, vbshvegsh), each (P, H, W).

    Bush-free path only: the caller guarantees bush.max()==0, so shadow()'s bush
    branch is omitted. Every other step is reproduced *exactly* — the integer
    slice+zero-fill `temp[dst]=src[src]-dz` becomes an integer gather
    `temp[p]=where(valid, a[r+dx_p, c+dy_p]-dz_p, 0)`, and the per-patch ragged
    march termination is handled with an active mask (required because vbshvegsh
    accumulates every step).

    pad_to: when set and > P, the batch is padded to exactly `pad_to` lanes by
    duplicating the last azimuth (outputs sliced back to P). Lanes are fully
    independent (per-lane gathers + elementwise ops only) and a duplicate lane
    has an identical termination column, so no real lane's output nor the march
    length can change; the only purpose is a FIXED (P,H,W) shape so the compiled
    step records one graph per tile instead of one per tail-batch size.
    """
    device = a.device
    H, W = a.shape[0], a.shape[1]
    P = int(azimuths.shape[0])

    deg = torch.pi / 180.0
    az = azimuths.to(device=device, dtype=torch.float32).clone()
    az = torch.where(az == 0.0, torch.full_like(az, 1e-12), az)   # match shadow(): 0 -> 1e-12
    P_real = P
    if pad_to is not None and int(pad_to) > P:
        az = torch.cat([az, az[-1:].expand(int(pad_to) - P)])
        P = int(pad_to)
    az_rad = az * deg
    tanaltbyscale = torch.tan(altitude * deg) / scale             # scalar tensor

    sinaz = torch.sin(az_rad); cosaz = torch.cos(az_rad); tanaz = torch.tan(az_rad)
    signsin = torch.sign(sinaz); signcos = torch.sign(cosaz)
    dssin = torch.abs(1.0 / sinaz); dscos = torch.abs(1.0 / cosaz)

    pibyfour = torch.pi / 4.0
    cond = ((az_rad >= pibyfour) & (az_rad < 3.0 * pibyfour)) | \
           ((az_rad >= 5.0 * pibyfour) & (az_rad < 7.0 * pibyfour))

    rows = torch.arange(H, device=device)
    cols = torch.arange(W, device=device)
    zero = torch.zeros((), device=device)
    one = torch.ones((), device=device)

    f = a.unsqueeze(0).expand(P, H, W).clone()                   # carried state (P,H,W)
    sh = torch.zeros((P, H, W), device=device)
    vegsh = torch.zeros((P, H, W), device=device)
    vbshvegsh = torch.zeros((P, H, W), device=device)

    # When elevations are non-negative (absolute DSM), a terminated patch's next-step core outputs
    # (nf/nsh/nvegsh) already equal its frozen values: temp<=0 <= f, so f/sh/vegsh don't move. Only
    # vbshvegsh (an accumulator) keeps drifting and needs the freeze mask. This lets us skip 3 of the
    # 4 per-step (P,H,W) torch.where masks. Bit-identical; falls back to the full freeze if a<0.
    # Set SOLWEIG_FAST_FREEZE=0 to force the original 4-way freeze (A/B benchmarking).
    fast_freeze = bool(a.min() >= 0) and os.environ.get("SOLWEIG_FAST_FREEZE", "1") == "1"

    max_steps = 2 * max(H, W) + 2                                # loop guard

    # Schedule-driven march (default): dx/dy/dz/active are closed-form in
    # (lane, step), so precompute the whole schedule with the SAME torch f32 ops
    # the while-loop uses (bit-identical per step, incl. `signsin * float(index)`
    # since every index <= max_steps is exact in f32) and derive the step count
    # with ONE host sync per batch instead of a bool(active.any()) sync per step.
    # Per-lane termination is prefix-monotone (dz strictly increases, |dxi|/|dyi|
    # non-decreasing), so `alive` rows are True-prefixed and the while-loop's
    # step count equals (# any-True rows) + 1, capped by the guard.
    # Set SOLWEIG_BATCH_SCHEDULE=0 to restore the stepwise loop (A/B benchmarking).
    if os.environ.get("SOLWEIG_BATCH_SCHEDULE", "1") == "1":
        k = torch.arange(1.0, float(max_steps) + 1.0, device=device, dtype=torch.float32)[:, None]
        dy_all = torch.where(cond, signsin * k, signsin * torch.abs(torch.round(k * tanaz)))
        dx_all = torch.where(cond, -1.0 * signcos * torch.abs(torch.round(k / tanaz)), -1.0 * signcos * k)
        ds = torch.where(cond, dssin, dscos)
        dz_all = ds * k * tanaltbyscale                          # (S,P), assoc matches ds*idxf*tanaltbyscale
        dxi_all = dx_all.round().to(torch.long)
        dyi_all = dy_all.round().to(torch.long)
        alive = (amaxvalue >= dz_all) & (dxi_all.abs() < H) & (dyi_all.abs() < W)
        S_run = min(int(alive.any(dim=1).sum().item()) + 1, max_steps)

        for index in range(1, S_run + 1):
            dxi = dxi_all[index - 1]
            dyi = dyi_all[index - 1]
            dz = dz_all[index - 1]
            if index == 1:
                # Eager first step: the firstvegdem block below needs temp and
                # tempvegdem, and the freeze mask is all-True at step 1 (a
                # torch.where with an all-True mask is a bitwise identity), so
                # core outputs are assigned directly.
                src_r = rows[None, :] + dxi[:, None]
                src_c = cols[None, :] + dyi[:, None]
                valid = ((src_r >= 0) & (src_r < H))[:, :, None] & ((src_c >= 0) & (src_c < W))[:, None, :]
                idx_r = src_r.clamp_(0, H - 1)[:, :, None]
                idx_c = src_c.clamp_(0, W - 1)[:, None, :]
                dz3 = dz[:, None, None]
                temp = torch.where(valid, a[idx_r, idx_c] - dz3, zero)
                tempvegdem = torch.where(valid, vegdem[idx_r, idx_c] - dz3, zero)
                tempvegdem2 = torch.where(valid, vegdem2[idx_r, idx_c] - dz3, zero)
                f, sh, _fab, vegsh, vbshvegsh = _shadow_core(
                    f, temp, a, tempvegdem, tempvegdem2, vegsh, vbshvegsh
                )
                firstvegdem = tempvegdem - temp
                firstvegdem = torch.where(firstvegdem <= 0.0, torch.full_like(firstvegdem, 1000.0), firstvegdem)
                vegsh = torch.where(firstvegdem < dz3, one, vegsh)
                vegsh = vegsh * (vegdem2 > a).float()
                vbshvegsh = torch.zeros_like(vbshvegsh)
                del temp, tempvegdem, tempvegdem2, firstvegdem
            elif fast_freeze:
                f, sh, vegsh, vbshvegsh = _batch_step_fast_dispatch(
                    f, vegsh, vbshvegsh, a, vegdem, vegdem2, rows, cols,
                    dxi, dyi, dz, alive[index - 2]
                )
            else:
                f, sh, vegsh, vbshvegsh = _batch_step_full_dispatch(
                    f, sh, vegsh, vbshvegsh, a, vegdem, vegdem2, rows, cols,
                    dxi, dyi, dz, alive[index - 2]
                )
    else:
        active = torch.ones(P, dtype=torch.bool, device=device)  # in-bounds at step 0
        index = 1
        while index <= max_steps and bool(active.any()):
            idxf = float(index)
            dy = torch.where(cond, signsin * idxf, signsin * torch.abs(torch.round(idxf * tanaz)))
            dx = torch.where(cond, -1.0 * signcos * torch.abs(torch.round(idxf / tanaz)), -1.0 * signcos * idxf)
            ds = torch.where(cond, dssin, dscos)
            dz = ds * idxf * tanaltbyscale                       # (P,)

            dxi = dx.round().to(torch.long)
            dyi = dy.round().to(torch.long)
            src_r = rows[None, :] + dxi[:, None]                 # (P,H)
            src_c = cols[None, :] + dyi[:, None]                 # (P,W)
            valid = ((src_r >= 0) & (src_r < H))[:, :, None] & ((src_c >= 0) & (src_c < W))[:, None, :]
            idx_r = src_r.clamp_(0, H - 1)[:, :, None]           # (P,H,1)
            idx_c = src_c.clamp_(0, W - 1)[:, None, :]           # (P,1,W)
            dz3 = dz[:, None, None]

            temp = torch.where(valid, a[idx_r, idx_c] - dz3, zero)   # (P,H,W), exact vs slice+zero_
            tempvegdem = torch.where(valid, vegdem[idx_r, idx_c] - dz3, zero)
            tempvegdem2 = torch.where(valid, vegdem2[idx_r, idx_c] - dz3, zero)

            nf, nsh, _fab, nvegsh, nvbsh = _shadow_core(f, temp, a, tempvegdem, tempvegdem2, vegsh, vbshvegsh)

            m = active[:, None, None]                            # freeze terminated patches
            if fast_freeze:
                # nf/nsh/nvegsh == frozen values for terminated patches (a>=0), so update in place and
                # only mask the accumulator.
                f, sh, vegsh = nf, nsh, nvegsh
                vbshvegsh = torch.where(m, nvbsh, vbshvegsh)
            else:
                f = torch.where(m, nf, f)
                sh = torch.where(m, nsh, sh)
                vegsh = torch.where(m, nvegsh, vegsh)
                vbshvegsh = torch.where(m, nvbsh, vbshvegsh)

            if index == 1:
                firstvegdem = tempvegdem - temp
                firstvegdem = torch.where(firstvegdem <= 0.0, torch.full_like(firstvegdem, 1000.0), firstvegdem)
                vegsh = torch.where(firstvegdem < dz3, one, vegsh)
                vegsh = vegsh * (vegdem2 > a).float()
                vbshvegsh = torch.zeros_like(vbshvegsh)

            active = (amaxvalue >= dz) & (dxi.abs() < H) & (dyi.abs() < W)   # in-bounds now -> active next
            index += 1

    sh = 1.0 - sh
    vbshvegsh = torch.where(vbshvegsh > 0.0, one, vbshvegsh)
    vbshvegsh = vbshvegsh - vegsh
    vegsh = torch.where(vegsh > 0.0, one, vegsh)
    vegsh = 1.0 - vegsh
    vbshvegsh = 1.0 - vbshvegsh
    return sh[:P_real], vegsh[:P_real], vbshvegsh[:P_real]


def annulus_weight(altitude, aziinterval, device=None):
    """
    Calculate annulus weights for sky view factor computation.
    
    Computes weights for different altitude bands used in SVF calculation
    based on the solid angle subtended by each annular ring.
    
    Args:
        altitude (float or torch.Tensor): Solar altitude angle (degrees)
        aziinterval (int): Azimuthal interval for discretization
        device (torch.device, optional): PyTorch device. Auto-detects if None.
    
    Returns:
        torch.Tensor: Array of annulus weights
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n = torch.tensor(90.0, device=device)
    altitude = torch.tensor(altitude, device=device)

    steprad = (360.0 / aziinterval) * (torch.pi / 180.0)
    annulus = 91.0 - altitude
    w = (1.0 / (2.0 * torch.pi)) * torch.sin(torch.pi / (2.0 * n)) * torch.sin((torch.pi * (2.0 * annulus - 1.0)) / (2.0 * n))
    weight = steprad * w

    return weight

def create_patches(patch_option):
    """
    Create patch configuration for sky hemisphere discretization.
    
    Generates the angular resolution and patch geometry for sky view factor
    calculations by dividing the sky hemisphere into discrete patches.
    
    Args:
        patch_option (int): Number of patches (144 or 2304)
            - 144: Coarser resolution (faster)
            - 2304: Finer resolution (more accurate)
    
    Returns:
        dict: Configuration containing:
            - 'azimuthinterval': Number of azimuth bins
            - 'altitudeinterval': Number of altitude bins
            - 'patchnorm': Normalization factor
    
    Raises:
        ValueError: If patch_option is not 144 or 2304
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deg2rad = torch.pi / 180
    skyvaultalt = torch.tensor([], device=device)
    skyvaultazi = torch.tensor([], device=device)

    if patch_option == 1:
        annulino = torch.tensor([0, 12, 24, 36, 48, 60, 72, 84, 90], device=device)
        skyvaultaltint = torch.tensor([6, 18, 30, 42, 54, 66, 78, 90], device=device)
        azistart = torch.tensor([0, 4, 2, 5, 8, 0, 10, 0], device=device)
        patches_in_band = torch.tensor([30, 30, 24, 24, 18, 12, 6, 1], device=device)
    elif patch_option == 2:
        annulino = torch.tensor([0, 12, 24, 36, 48, 60, 72, 84, 90], device=device)
        skyvaultaltint = torch.tensor([6, 18, 30, 42, 54, 66, 78, 90], device=device)
        azistart = torch.tensor([0, 4, 2, 5, 8, 0, 10, 0], device=device)
        patches_in_band = torch.tensor([31, 30, 28, 24, 19, 13, 7, 1], device=device)
    elif patch_option == 3:
        annulino = torch.tensor([0, 12, 24, 36, 48, 60, 72, 84, 90], device=device)
        skyvaultaltint = torch.tensor([6, 18, 30, 42, 54, 66, 78, 90], device=device)
        azistart = torch.tensor([0, 4, 2, 5, 8, 0, 10, 0], device=device)
        patches_in_band = torch.tensor([31*2, 30*2, 28*2, 24*2, 19*2, 13*2, 7*2, 1], device=device)
    elif patch_option == 4:
        annulino = torch.tensor([0, 4.5, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90], device=device)
        skyvaultaltint = torch.tensor([3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90], device=device)
        patches_in_band = torch.tensor([31*2, 31*2, 30*2, 30*2, 28*2, 28*2, 24*2, 24*2, 19*2, 19*2, 13*2, 13*2, 7*2, 7*2, 1], device=device)
        azistart = torch.tensor([0, 0, 4, 4, 2, 2, 5, 5, 8, 8, 0, 0, 10, 10, 0], device=device)

    skyvaultaziint = 360 / patches_in_band

    for j in range(skyvaultaltint.shape[0]):
        for k in range(patches_in_band[j]):
            skyvaultalt = torch.cat((skyvaultalt, torch.tensor([skyvaultaltint[j]], device=device)))
            skyvaultazi = torch.cat((skyvaultazi, torch.tensor([k * skyvaultaziint[j] + azistart[j]], device=device)))

    return skyvaultalt, skyvaultazi, annulino, skyvaultaltint, patches_in_band, skyvaultaziint, azistart


def svf_calculator(patch_option, amaxvalue=None, a=None, vegdem=None, vegdem2=None, bush=None, scale=None, save_rasters=False,
    building_dsm_path=None, tree_path=None, dem_path=None, output_dir=None, number=None, gdal_dsm=None,):
    """
    Calculate Sky View Factor (SVF) using GPU-accelerated hemisphere sampling.
    
    SVF represents the portion of visible sky from each point, accounting for
    obstructions from buildings and vegetation. Directional SVFs are also computed
    for cardinal directions (N, E, S, W).
    
    Args:
        patch_option (int): Sky discretization option (144 or 2304 patches)
        amaxvalue (torch.Tensor): Maximum elevation in domain
        a (torch.Tensor): Digital Surface Model
        vegdem (torch.Tensor): Vegetation canopy DSM
        vegdem2 (torch.Tensor): Vegetation trunk zone DSM
        bush (torch.Tensor): Bush layer DSM
        scale (float): Grid resolution (pixels per meter)
    
    Returns:
        tuple: (svf, svfE, svfS, svfW, svfN, svfveg, svfEveg, svfSveg, svfWveg, svfNveg, 
                svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg) where:
            - svf: Total sky view factor [0-1]
            - svfE/S/W/N: Directional SVFs for East/South/West/North
            - svf*veg: Vegetation-obstructed SVFs
            - svf*aveg: Vegetation-adjusted SVFs
    
    Notes:
        - Uses GPU if available for fast computation
        - Higher patch_option gives more accurate but slower results
        - Directional SVFs useful for anisotropic radiation modeling
    """

    gdal_dsm_local = gdal_dsm

    if save_rasters:
        if output_dir is None:
            raise ValueError("When save_rasters=True, output_dir must be provided.")

        # Standalone mode: compute missing SVF inputs from raster paths
        if any(x is None for x in [amaxvalue, a, vegdem, vegdem2, bush, scale]):
            if building_dsm_path is None or tree_path is None or dem_path is None:
                raise ValueError(
                    "When save_rasters=True and SVF tensors are not provided, "
                    "building_dsm_path, tree_path, and dem_path must be provided."
                )

            a, gdal_dsm_local = load_raster_to_tensor(building_dsm_path)
            temp1, _ = load_raster_to_tensor(tree_path)
            temp2, _ = load_raster_to_tensor(dem_path)

            geotransform = gdal_dsm_local.GetGeoTransform()
            scale = 1.0 / geotransform[1]

            temp1[temp1 < 0.0] = 0.0

            # Same logic as compute_utci()
            vegdem_height = temp1 + temp2
            vegdem2_height = torch.add(temp1 * 0.25, temp2)

            bush = torch.logical_not(vegdem2_height * vegdem_height) * vegdem_height

            vegdem = temp1 + a
            vegdem[vegdem == a] = 0.0

            vegdem2 = temp1 * 0.25 + a
            vegdem2[vegdem2 == a] = 0.0

            amaxvalue = torch.maximum(a.max(), vegdem_height.max())

        else:
            if gdal_dsm_local is None:
                if building_dsm_path is not None:
                    gdal_dsm_local = gdal.Open(building_dsm_path)
                else:
                    raise ValueError(
                        "When save_rasters=True with precomputed tensors, provide "
                        "either gdal_dsm or building_dsm_path so GeoTIFF metadata can be copied."
                    )

    else:
        required = {
            "amaxvalue": amaxvalue,
            "a": a,
            "vegdem": vegdem,
            "vegdem2": vegdem2,
            "bush": bush,
            "scale": scale,
        }

        missing = [name for name, value in required.items() if value is None]

        if missing:
            raise ValueError(
                "When save_rasters=False, these arguments must be provided: "
                + ", ".join(missing)
            )
        
    device = a.device
    rows = a.shape[0]
    cols = a.shape[1]

    # Termination ceiling: the march only needs to span the true surface height RANGE. Once dz
    # exceeds it, every remaining step is a provable no-op for the (binarized) shadow outputs, so
    # capping the loop there is bit-identical while dropping the long tail of wasted steps that the
    # original `amaxvalue` (an ABSOLUTE elevation) marched through. The maximum() guard stays exact
    # even if a building sits under canopy (vegdem.max > amaxvalue). Set SOLWEIG_MARCH_CEILING=0 to
    # restore the original absolute ceiling (A/B benchmarking). Elevations themselves are untouched.
    hmin = a.min()
    if os.environ.get("SOLWEIG_MARCH_CEILING", "1") == "1":
        march_ceiling = torch.maximum(amaxvalue, vegdem.max()) - hmin
    else:
        march_ceiling = amaxvalue

    svf = torch.zeros([rows, cols], device=device)
    svfE = torch.zeros([rows, cols], device=device)
    svfS = torch.zeros([rows, cols], device=device)
    svfW = torch.zeros([rows, cols], device=device)
    svfN = torch.zeros([rows, cols], device=device)
    svfveg = torch.zeros((rows, cols), device=device)
    svfEveg = torch.zeros((rows, cols), device=device)
    svfSveg = torch.zeros((rows, cols), device=device)
    svfWveg = torch.zeros((rows, cols), device=device)
    svfNveg = torch.zeros((rows, cols), device=device)
    svfaveg = torch.zeros((rows, cols), device=device)
    svfEaveg = torch.zeros((rows, cols), device=device)
    svfSaveg = torch.zeros((rows, cols), device=device)
    svfWaveg = torch.zeros((rows, cols), device=device)
    svfNaveg = torch.zeros((rows, cols), device=device)

    skyvaultalt, skyvaultazi, annulino, skyvaultaltint, aziinterval, skyvaultaziint, azistart = create_patches(patch_option)
    skyvaultaziint = torch.tensor([360 / patches for patches in aziinterval], device=device)
    iazimuth = torch.zeros((1, torch.sum(aziinterval).item()), device=device)

    shmat = torch.zeros((rows, cols, sum(aziinterval)), device=device)
    vegshmat = torch.zeros((rows, cols, sum(aziinterval)), device=device)
    vbshvegshmat = torch.zeros((rows, cols, sum(aziinterval)), device=device)

    index = 0
    for j in range(skyvaultaltint.shape[0]):
        for k in range(int(360 / skyvaultaziint[j])):
            iazimuth[0, index] = k * skyvaultaziint[j] + azistart[j]
            if iazimuth[0, index] > 360.:
                iazimuth[0, index] = iazimuth[0, index] - 360.
            index += 1

    aziintervalaniso = torch.ceil(aziinterval / 2.0)

    # Precompute per-altitude-band annulus weight sums once. annulus_weight(k, .)
    # is a pixel-independent scalar, so summing weight*sh over the ring index k is
    # (sum_k weight) * sh -- collapsing the inner per-patch loop into a few adds.
    n_bands = skyvaultaltint.shape[0]
    Wfull = [None] * n_bands
    Waniso = [None] * n_bands
    for i in range(n_bands):
        wf = torch.zeros((), device=device)
        wa = torch.zeros((), device=device)
        for k in range(int(annulino[i]) + 1, int(annulino[i + 1]) + 1):
            wf = wf + annulus_weight(k, aziinterval[i], device)
            wa = wa + annulus_weight(k, aziintervalaniso[i], device)
        Wfull[i] = wf
        Waniso[i] = wa

    # Patch-batched march: fill shmat/vegshmat/vbshvegshmat one altitude band at a
    # time (all its azimuths at once) instead of one patch at a time. Bush-free only;
    # bush tiles or SOLWEIG_BATCH_MARCH=0 fall back to the per-patch path below.
    use_batch = (os.environ.get("SOLWEIG_BATCH_MARCH", "1") == "1") and (not bool(bush.max() > 0.))
    if use_batch:
        try:
            _bsz = int(os.environ.get("SOLWEIG_BATCH_SIZE", "16"))
        except ValueError:
            _bsz = 16
        # VRAM-aware clamp: a march step holds ~14 live f32 planes per lane
        # (carried state, gather outputs, core temporaries); cap the lane count
        # to half the free headroom so big tiles can't OOM. Partitioning cannot
        # change any lane's output (lanes are independent), only throughput.
        # Set SOLWEIG_BATCH_CLAMP=0 to use SOLWEIG_BATCH_SIZE verbatim.
        if torch.cuda.is_available() and os.environ.get("SOLWEIG_BATCH_CLAMP", "1") == "1":
            try:
                free_b, _total_b = torch.cuda.mem_get_info()
                pool_slack = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                per_lane = rows * cols * 4 * 14
                cap = max(1, int((free_b + pool_slack) * 0.5 // per_lane))
                if cap < _bsz:
                    print(f"[SOLWEIG] SOLWEIG_BATCH_SIZE {_bsz} -> {cap} (VRAM headroom)")
                    _bsz = cap
            except Exception:  # pragma: no cover - never block the march on accounting
                pass
        # Optionally pad tail batches to a fixed lane count so the compiled step
        # sees one (P,H,W) shape per tile instead of ~9 (see shadow_batch
        # docstring). Off by default: padded lanes cost real bandwidth, and the
        # zenith band (altitude 90, P=1) is pathological — float32 tan(90 deg) is
        # NEGATIVE, so its dz never exceeds the ceiling and it marches to the
        # grid bound (~max(H,W) steps) in both this and the original code;
        # padding that 1-lane batch 16x multiplies the waste. Per-shape
        # recompiles are cheap (dynamo cache limit is raised alongside the
        # compiled step) and happen once per process.
        _pad_to = _bsz if os.environ.get("SOLWEIG_BATCH_PAD", "0") == "1" else None
        _gstart = 0
        for i in range(skyvaultaltint.shape[0]):
            Pi = int(aziinterval[i])
            altitude_i = skyvaultaltint[i]
            band_az = iazimuth[0, _gstart:_gstart + Pi]
            c = 0
            while c < Pi:
                cP = min(_bsz, Pi - c)
                shb, vegshb, vbshb = shadow_batch(
                    march_ceiling, a, vegdem, vegdem2, band_az[c:c + cP], altitude_i, scale,
                    pad_to=_pad_to
                )
                s = _gstart + c
                shmat[:, :, s:s + cP] = shb.permute(1, 2, 0)
                vegshmat[:, :, s:s + cP] = vegshb.permute(1, 2, 0)
                vbshvegshmat[:, :, s:s + cP] = vbshb.permute(1, 2, 0)
                del shb, vegshb, vbshb
                c += cP
            _gstart += Pi

    # Read the per-patch azimuths to host once: the direction tests below
    # (`0 <= azimuth < 180` etc.) on 0-dim GPU tensors would otherwise cost
    # ~1200 GPU->CPU syncs per tile. float32 -> Python float is exact, so every
    # branch decision (and hence accumulation order and operands) is unchanged.
    # Set SOLWEIG_FAST_ACCUM=0 to restore the tensor comparisons.
    fast_accum = os.environ.get("SOLWEIG_FAST_ACCUM", "1") == "1"
    iaz_host = iazimuth[0].detach().cpu().tolist() if fast_accum else None

    index = 0
    for i in range(skyvaultaltint.shape[0]):
        for j in range(aziinterval[i].int()):
            altitude = skyvaultaltint[i]
            azimuth_t = iazimuth[0, index]
            azimuth = iaz_host[index] if fast_accum else azimuth_t
            if use_batch:
                sh = shmat[:, :, index]
                vegsh = vegshmat[:, :, index]
                vbshvegsh = vbshvegshmat[:, :, index]
            else:
                sh, vegsh, vbshvegsh = shadow(march_ceiling, a, vegdem, vegdem2, bush, azimuth_t, altitude, scale)
                vegshmat[:, :, index] = vegsh
                vbshvegshmat[:, :, index] = vbshvegsh
                shmat[:, :, index] = sh

            wf = Wfull[i]
            wa = Waniso[i]
            svf = svf + wf * sh
            weight = wa * sh
            if 0 <= azimuth < 180:
                svfE = svfE + weight
            if 90 <= azimuth < 270:
                svfS = svfS + weight
            if 180 <= azimuth < 360:
                svfW = svfW + weight
            if azimuth >= 270 or azimuth < 90:
                svfN = svfN + weight

            svfveg = svfveg + wf * vegsh
            svfaveg = svfaveg + wf * vbshvegsh
            if 0 <= azimuth < 180:
                svfEveg = svfEveg + wa * vegsh
                svfEaveg = svfEaveg + wa * vbshvegsh
            if 90 <= azimuth < 270:
                svfSveg = svfSveg + wa * vegsh
                svfSaveg = svfSaveg + wa * vbshvegsh
            if 180 <= azimuth < 360:
                svfWveg = svfWveg + wa * vegsh
                svfWaveg = svfWaveg + wa * vbshvegsh
            if azimuth >= 270 or azimuth < 90:
                svfNveg = svfNveg + wa * vegsh
                svfNaveg = svfNaveg + wa * vbshvegsh

            index += 1
    svfS = svfS + 3.0459e-004
    svfW = svfW + 3.0459e-004
    svf[svf > 1.] = 1.
    svfE[svfE > 1.] = 1.
    svfS[svfS > 1.] = 1.
    svfW[svfW > 1.] = 1.
    svfN[svfN > 1.] = 1.

    last = torch.zeros((rows, cols), device=device)
    last[vegdem2 == 0.] = 3.0459e-004
    svfSveg = svfSveg + last
    svfWveg = svfWveg + last
    svfSaveg = svfSaveg + last
    svfWaveg = svfWaveg + last
    svfveg[svfveg > 1.] = 1.
    svfEveg[svfEveg > 1.] = 1.
    svfSveg[svfSveg > 1.] = 1.
    svfWveg[svfWveg > 1.] = 1.
    svfNveg[svfNveg > 1.] = 1.
    svfaveg[svfaveg > 1.] = 1.
    svfEaveg[svfEaveg > 1.] = 1.
    svfSaveg[svfSaveg > 1.] = 1.
    svfWaveg[svfWaveg > 1.] = 1.
    svfNaveg[svfNaveg > 1.] = 1.

    trans = torch.tensor(0.03, device=device)  # Tree transmission hardcoded to 3%
    SVFtotal = svf - (1 - svfveg) * (1 - trans)

    if save_rasters:
        save_svf_zip_npz_outputs(output_dir=output_dir, gdal_dsm=gdal_dsm_local, svf=svf, svfE=svfE, svfS=svfS, svfW=svfW, svfN=svfN,
            svfveg=svfveg, svfEveg=svfEveg, svfSveg=svfSveg, svfWveg=svfWveg, svfNveg=svfNveg, svfaveg=svfaveg, svfEaveg=svfEaveg,
            svfSaveg=svfSaveg, svfWaveg=svfWaveg, svfNaveg=svfNaveg, shmat=shmat, vegshmat=vegshmat, vbshvegshmat=vbshvegshmat,
            svftotal=SVFtotal, number=number,
        )

    del sh, vegsh, vbshvegsh, last, weight
    return svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg, svfNveg, svfS, svfSaveg, svfSveg, svfveg, svfW, svfWaveg, svfWveg, vegshmat, vbshvegshmat, shmat, SVFtotal

