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
import os
import numpy as np
from osgeo import gdal
from scipy.ndimage import rotate, maximum_filter, correlate
import math
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import gc

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is a hard dependency in practice
    _TORCH_AVAILABLE = False

gdal.UseExceptions()

# Wall height threshold
walllimit = 3.0

def findwalls(dem_array, walllimit):
    """
    Identify walls in a Digital Surface Model (DSM) based on height threshold.
    
    Walls are detected by comparing each cell to its immediate neighbors.
    A wall exists where the elevation difference exceeds the threshold.
    
    Args:
        dem_array (np.ndarray): 2D array of elevation values (DSM)
        walllimit (float): Minimum height difference (m) to be considered a wall
    
    Returns:
        np.ndarray: 2D array of wall heights. Zero where no wall exists.
    """
    # Wall height at each cell = (max of its 4 orthogonal neighbours) - itself.
    # This is a maximum filter over the plus-shaped neighbourhood, so we use
    # scipy's C-level maximum_filter instead of a per-pixel Python loop. The
    # result is identical on the interior (the border is zeroed explicitly
    # below, exactly as before) and is ~2-3 orders of magnitude faster.
    domain = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    walls = maximum_filter(dem_array, footprint=domain, mode='nearest') - dem_array
    walls[walls < walllimit] = 0

    walls[:, 0] = 0
    walls[:, -1] = 0
    walls[0, :] = 0
    walls[-1, :] = 0

    return walls

def cart2pol(x, y, units='deg'):
    """
    Convert Cartesian coordinates to polar coordinates.
    
    Args:
        x (np.ndarray or float): X coordinate(s)
        y (np.ndarray or float): Y coordinate(s)
        units (str): Output angle units ('deg' or 'rad'). Default: 'deg'
    
    Returns:
        tuple: (theta, radius) where theta is angle and radius is distance
    """
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if units in ['deg', 'degs']:
        theta = theta * 180 / np.pi
    return theta, radius

def get_ders(dsm, scale):
    """
    Calculate slope derivatives (aspect and gradient) from DSM.
    
    Args:
        dsm (np.ndarray): Digital Surface Model array
        scale (float): Pixel size in meters
    
    Returns:
        tuple: (aspect, gradient) where:
            - aspect: slope orientation in radians
            - gradient: slope magnitude
    """
    dx = 1 / scale
    fy, fx = np.gradient(dsm, dx, dx)
    asp, grad = cart2pol(fy, fx, 'rad')
    grad = np.arctan(grad)
    asp = -asp
    asp[asp < 0] += 2 * np.pi
    return grad, asp

@lru_cache(maxsize=8)
def _rotated_aspect_filters(scale):
    """
    Precompute the 180 rotated directional filters used for wall aspect.

    The filters depend only on ``scale`` (through the derived ``filtersize``),
    not on the DSM data, so they are identical for every tile of a given raster.
    Building them once and caching avoids recomputing 360 rotations per tile.

    Returns:
        tuple: ``(f1_stack, b1_stack, b2_stack, index_arr, filthalveceil)``
        where the three stacks have shape ``(180, filtersize, filtersize)``.
    """
    filtersize = int(np.floor((scale + 1e-10) * 9))
    if filtersize <= 2:
        filtersize = 3
    elif filtersize != 9 and filtersize % 2 == 0:
        filtersize += 1

    n = filtersize - 1
    filthalveceil = int(np.ceil(filtersize / 2.))
    filthalvefloor = int(np.floor(filtersize / 2.))

    filtmatrix = np.zeros((filtersize, filtersize))
    buildfilt = np.zeros((filtersize, filtersize))
    filtmatrix[:, filthalveceil - 1] = 1
    buildfilt[filthalveceil - 1, :filthalvefloor] = 1
    buildfilt[filthalveceil - 1, filthalveceil:] = 2

    f1_stack, b1_stack, b2_stack, index_arr = [], [], [], []
    for h in range(0, 180):
        filtmatrix1 = np.round(rotate(filtmatrix, h, order=1, reshape=False, mode='nearest'))
        filtmatrixbuild = np.round(rotate(buildfilt, h, order=0, reshape=False, mode='nearest'))
        index = 270 - h

        if h in [150, 30]:
            filtmatrixbuild[:, n] = 0
        if index == 225:
            filtmatrix1[0, 0] = filtmatrix1[n, n] = 1
        if index == 135:
            filtmatrix1[0, n] = filtmatrix1[n, 0] = 1

        f1_stack.append(filtmatrix1)
        b1_stack.append((filtmatrixbuild == 1).astype(np.float32))
        b2_stack.append((filtmatrixbuild == 2).astype(np.float32))
        index_arr.append(index)

    return (np.asarray(f1_stack, dtype=np.float32),
            np.asarray(b1_stack, dtype=np.float32),
            np.asarray(b2_stack, dtype=np.float32),
            np.asarray(index_arr, dtype=np.float32),
            filthalveceil)


def _side_offsets(mask, filthalvefloor):
    """
    ``(dr, dc)`` offsets of the cells of one build-filter side, in the C (row-
    major) order that ``np.sum(dsm_window[mask])`` visits them. Matching this
    order lets us reproduce the reference's float32 summation exactly.
    """
    fs = mask.shape[0]
    return [(r - filthalvefloor, c - filthalvefloor)
            for r in range(fs) for c in range(fs) if mask[r, c] > 0]


def _winner_angle_map(walls, f1_stack, filthalveceil, use_gpu):
    """
    For every interior wall pixel, find the rotation angle whose line filter
    overlaps the most wall cells. Returns an int array holding the winning angle
    index per pixel, or ``-1`` where there is no wall or no overlap at any angle
    (mirroring the reference's ``z < s`` guard, which never fires when every
    angle gives 0).

    The wall-count correlation of a 0/1 mask with a 0/1 kernel is an exact small
    integer even in float32, so GPU and CPU yield identical winner maps, and the
    running strict-``>`` update keeps the first (lowest-angle) winner exactly as
    the reference loop does.
    """
    row, col = walls.shape
    fs = f1_stack.shape[-1]
    n_ang = f1_stack.shape[0]
    wbin = (walls > 0)

    if use_gpu:
        device = 'cuda'
        pad = fs // 2  # filtersize is always odd -> symmetric 'same' padding
        wt = torch.as_tensor(wbin, dtype=torch.float32, device=device)[None, None]
        kF1 = torch.as_tensor(f1_stack, device=device)[:, None]
        z = torch.zeros((row, col), device=device)
        win = torch.full((row, col), -1, dtype=torch.int64, device=device)
        chunk = 16
        while True:
            try:
                for s0 in range(0, n_ang, chunk):
                    s1 = min(s0 + chunk, n_ang)
                    wsum = F.conv2d(wt, kF1[s0:s1], padding=pad)[0]
                    for c in range(s1 - s0):
                        upd = wsum[c] > z
                        z[upd] = wsum[c][upd]
                        win[upd] = s0 + c
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if chunk == 1:
                    raise
                chunk //= 2
                z.zero_()
                win.fill_(-1)
        win = win.cpu().numpy()
    else:
        wbin_f = wbin.astype(np.float32)
        z = np.zeros((row, col), dtype=np.float32)
        win = np.full((row, col), -1, dtype=np.int64)
        for h in range(n_ang):
            wsum = correlate(wbin_f, f1_stack[h], mode='constant', cval=0.0)
            upd = wsum > z
            z[upd] = wsum[upd]
            win[upd] = h

    # The reference loop only ever visited interior wall pixels.
    keep = np.zeros((row, col), dtype=bool)
    keep[filthalveceil - 1:row - filthalveceil - 1,
         filthalveceil - 1:col - filthalveceil - 1] = True
    keep &= wbin
    win[~keep] = -1
    return win


def _resolve_sides_exact(a, win, b1_stack, b2_stack, filthalveceil):
    """
    Decide the building side (``x`` = 1 or 2) for each winning wall pixel.

    This reproduces the reference bit-for-bit: for a pixel's winning angle it
    compares ``sum(dsm over the '1' cells)`` against ``sum(dsm over the '2'
    cells)`` using the same float32 ``np.sum`` reduction order. Doing the sums
    this way (rather than via conv/correlate) is what keeps genuinely near-tied
    pixels from flipping 180 degrees, since any other summation order rounds a
    near-tie the other way.
    """
    row, col = a.shape
    fhf = b1_stack.shape[-1] // 2
    x = np.zeros((row, col))
    for h in np.unique(win[win >= 0]):
        rows, cols = np.where(win == h)
        o1 = _side_offsets(b1_stack[h], fhf)
        o2 = _side_offsets(b2_stack[h], fhf)
        # np.stack(..., axis=1) -> (n_pixels, n_cells), C-contiguous, so the
        # inner-axis np.sum uses the same reduction as np.sum(dsm_window[mask]).
        b1 = (np.sum(np.stack([a[rows + dr, cols + dc] for dr, dc in o1], axis=1), axis=1)
              if o1 else np.zeros(rows.size, dtype=a.dtype))
        b2 = (np.sum(np.stack([a[rows + dr, cols + dc] for dr, dc in o2], axis=1), axis=1)
              if o2 else np.zeros(rows.size, dtype=a.dtype))
        x[rows, cols] = np.where(b1 > b2, 1.0, 2.0)
    return x


def filter1Goodwin_as_aspect_v3(walls, scale, a):
    """
    Calculate wall aspect (orientation) using directional filtering.

    This function determines the orientation of walls by rotating a directional
    filter and finding the direction with maximum wall presence.

    Args:
        walls (np.ndarray): Binary array indicating wall locations
        scale (float): Pixel size in meters
        a (np.ndarray): Aspect array from DSM derivatives

    Returns:
        np.ndarray: Wall aspect in degrees [0-360], where 0=North, 90=East, 180=South, 270=West

    Notes:
        The reference's per-angle, per-pixel loop is split into two exact,
        vectorised steps: (1) a winning-angle argmax via wall-count correlation
        (batched ``conv2d`` on the GPU when CUDA is available, else
        ``scipy.ndimage`` on the CPU), and (2) the building-side decision, which
        is resolved with the reference's own float32 ``np.sum`` over the winning
        angle's cells. Both steps reproduce the original output bit-for-bit.
    """
    f1_stack, b1_stack, b2_stack, index_arr, filthalveceil = \
        _rotated_aspect_filters(float(scale))

    use_gpu = _TORCH_AVAILABLE and torch.cuda.is_available()
    win = _winner_angle_map(walls, f1_stack, filthalveceil, use_gpu)
    x = _resolve_sides_exact(a, win, b1_stack, b2_stack, filthalveceil)

    y = np.where(win >= 0, index_arr[np.maximum(win, 0)], 0.0).astype(np.float64)
    y[x == 1] -= 180
    y[y < 0] += 360

    grad, asp = get_ders(a, scale)
    y += ((walls > 0) & (y == 0)) * (asp / (math.pi / 180.))

    return y

def _tile_key_from_dsm(filename):
    """
    Extract the tile key from a Building_DSM tile filename.

    The prefix ``Building_DSM_`` is 13 characters, so e.g.
    ``Building_DSM_0_0.tif`` -> ``0_0``.
    """
    return filename[13:-4]

def process_file_parallel(args):
    """
    Process a single DEM tile to calculate walls and aspect (parallel worker function).

    This function is designed to be called by parallel processing workers.

    Args:
        args (tuple): (filename, dem_folder_path, wall_output_path,
            aspect_output_path, overwrite)

    Returns:
        str: Filename of processed tile
    """
    filename, dem_folder_path, wall_output_path, aspect_output_path, overwrite = args
    dem_path = os.path.join(dem_folder_path, filename)

    # Walls and aspect depend only on geometry (not time), so skip tiles that
    # already have both outputs unless an overwrite was explicitly requested.
    key = _tile_key_from_dsm(filename)
    wall_out_path = os.path.join(wall_output_path, f"walls_{key}.tif")
    aspect_out_path = os.path.join(aspect_output_path, f"aspect_{key}.tif")

    if (not overwrite) and os.path.exists(wall_out_path) and os.path.exists(aspect_out_path):
        return filename

    dataset = None
    try:
        dataset = gdal.Open(dem_path)
        if dataset is None:
            print(f"Could not open {filename}")
            return filename

        band = dataset.GetRasterBand(1)
        a = band.ReadAsArray().astype(np.float32)

        if a is None or np.all(np.isnan(a)):
            print(f"Skipping {filename}, invalid DEM.")
            # Close dataset before returning
            dataset = None
            return filename

        # Extract geotransform and projection before closing dataset
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        
        # Close input dataset immediately after reading
        dataset = None
        
        scale = 1 / geotransform[1]
        walls = findwalls(a, walllimit)
        aspects = filter1Goodwin_as_aspect_v3(walls, scale, a)

        driver = gdal.GetDriverByName('GTiff')
        out_paths = [wall_out_path, aspect_out_path]

        for out_path, data in zip(out_paths, [walls, aspects]):
            # Retry logic for Windows file locking issues
            max_retries = 3
            retry_delay = 0.1
            success = False
            
            for attempt in range(max_retries):
                try:
                    out_ds = driver.Create(out_path, a.shape[1], a.shape[0], 1, gdal.GDT_Float32)
                    if out_ds is None:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                        print(f"Could not create output file {out_path}")
                        break
                        
                    out_ds.SetGeoTransform(geotransform)
                    out_ds.SetProjection(projection)
                    out_band = out_ds.GetRasterBand(1)
                    out_band.WriteArray(data)
                    out_band.FlushCache()
                    out_band = None
                    
                    # Explicitly close the dataset
                    out_ds.FlushCache()
                    out_ds = None
                    
                    # Force GDAL to close the file handle on Windows
                    gc.collect()
                    
                    # Small delay to ensure file handle is released
                    time.sleep(0.01)
                    success = True
                    break
                    
                except Exception as e:
                    # Ensure cleanup before retry
                    out_ds = None
                    gc.collect()
                    
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        print(f"Error writing {out_path} after {max_retries} attempts: {e}")
                        break

        return filename

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        # Ensure dataset is closed even on error
        if dataset is not None:
            dataset = None
        return filename

def run_parallel_processing(dem_folder_path, wall_output_path, aspect_output_path, overwrite=False):
    """
    Process all DEM tiles in parallel to calculate walls and aspects.

    This is the main entry point for wall and aspect calculation. It uses
    multiprocessing to process multiple tiles simultaneously for efficiency.

    Args:
        dem_folder_path (str): Path to folder containing DEM tile GeoTIFFs
        wall_output_path (str): Output path for wall height rasters
        aspect_output_path (str): Output path for wall aspect rasters
        overwrite (bool): If False (default), tiles that already have both a
            walls and an aspect output are skipped and reused. Set True to force
            recomputation (e.g. after changing tile_size/overlap or the inputs).

    Notes:
        - Uses multiple CPU cores for parallel processing
        - On Windows, uses fewer workers (max 8 or half of CPU cores) to avoid file locking issues
        - Progress bar shows processing status
        - Creates output directories if they don't exist
        - Skips tiles that cannot be opened or have invalid data
    """
    os.makedirs(wall_output_path, exist_ok=True)
    os.makedirs(aspect_output_path, exist_ok=True)

    dem_files = [f for f in os.listdir(dem_folder_path) if f.endswith('.tif') and not f.startswith('.')]

    if not overwrite:
        reused = sum(
            1 for f in dem_files
            if os.path.exists(os.path.join(wall_output_path, f"walls_{_tile_key_from_dsm(f)}.tif"))
            and os.path.exists(os.path.join(aspect_output_path, f"aspect_{_tile_key_from_dsm(f)}.tif"))
        )
        if reused:
            print(f"Reusing {reused}/{len(dem_files)} existing walls/aspect tiles (overwrite=False)")

    args_list = [(f, dem_folder_path, wall_output_path, aspect_output_path, overwrite) for f in dem_files]

    # When a CUDA GPU is available the aspect step runs on it (batched conv2d).
    # A single process must own the device -- many workers contending for one
    # GPU is slower and can OOM -- and each tile now takes only a few seconds,
    # so tiles are processed sequentially in-process.
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        print("Computing Wall Height and Aspect on GPU (sequential over tiles)")
        for args in tqdm(args_list, total=len(args_list), desc="Computing Wall Height and Aspect", mininterval=1):
            process_file_parallel(args)
        return

    # CPU fallback: the aspect step is vectorised (scipy correlate) but still
    # CPU-bound, so fan tiles out across processes as before.
    # On Windows, use fewer workers to avoid file locking issues
    # Reduce max_workers to prevent excessive concurrent file access
    cpu_count = os.cpu_count() or 1
    if os.name == 'nt':  # Windows
        max_workers = min(8, max(1, cpu_count // 2))
    else:
        max_workers = min(32, cpu_count)
    print(f"Using {max_workers} parallel workers")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file_parallel, args): args[0] for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing Wall Height and Aspect", mininterval = 1):
            _ = future.result()

