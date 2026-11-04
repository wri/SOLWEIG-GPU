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
"""
Post-processing helpers for SOLWEIG-GPU outputs.

:func:`mosaic_tiles` stitches the per-tile GeoTIFFs written by
:func:`solweig_gpu.run_utci_tiles` back into a single raster per variable.
"""
from typing import List, Optional


def _band_filename(var: str, time_str: Optional[str], band_index: int) -> str:
    """
    Name a single-band (per-hour) mosaic file from its band ``Time`` metadata.

    ``2024-05-27T09:30:00`` -> ``UTCI_2024-05-27_0930.tif``. Falls back to a
    1-based band number (e.g. ``SVF_band01.tif``) when no timestamp is present.
    """
    if time_str and "T" in time_str:
        date_part, time_part = time_str.split("T", 1)
        hhmm = time_part.replace(":", "")[:4]
        return f"{var}_{date_part}_{hhmm}.tif"
    return f"{var}_band{band_index + 1:02d}.tif"


def mosaic_tiles(
    base_path: str,
    tile_size: int,
    variables: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    overwrite: bool = True,
    compress: Optional[str] = "DEFLATE",
    split_bands: bool = False,
) -> str:
    """
    Mosaic the per-tile SOLWEIG-GPU output GeoTIFFs into one raster per variable.

    :func:`run_utci_tiles` writes results per tile to
    ``{base_path}/output_folder/{i}_{j}/{VAR}_{i}_{j}.tif`` (multi-band, one band
    per timestep, each band carrying a ``Time`` metadata item). This reassembles
    them into ``{base_path}/output_folder/mosaic/{VAR}.tif``.

    Tiles are cut with a right/bottom-only overlap (see
    :func:`solweig_gpu.preprocessor.create_tiles`); the overlap is context for
    shadow transfer and is discarded here. Each tile contributes only its
    top-left ``tile_size x tile_size`` core (last row/column keep their full
    extent), so the mosaic reproduces the original raster grid exactly.

    Args:
        base_path: Base directory; ``output_folder`` lives under it.
        tile_size: Tile size in pixels used at preprocessing (the same value
            passed to :func:`preprocess`). Used to trim the overlap fringe.
        variables: Which variables to mosaic (e.g. ``["UTCI", "TMRT"]``). If
            None, every variable found under the tile folders is mosaicked.
        output_dir: Where to write mosaics. Defaults to
            ``{base_path}/output_folder/mosaic``.
        overwrite: If True, replace existing mosaic files; else skip them.
        compress: GeoTIFF compression (e.g. ``"DEFLATE"``, ``"LZW"``). Falsy
            disables compression.
        split_bands: If True, write **one single-band GeoTIFF per band** (per
            hour), named from each band's ``Time`` (e.g. ``UTCI_2024-05-27_0930.tif``),
            and do **not** write the multi-band file. If False (default), write one
            multi-band GeoTIFF per variable (all timesteps as bands).

    Returns:
        The directory the mosaics were written to.
    """
    import os
    from osgeo import gdal

    gdal.UseExceptions()

    NAN = float("nan")

    out_root = os.path.join(base_path, "output_folder")
    if not os.path.isdir(out_root):
        raise FileNotFoundError(f"Output folder not found: {out_root}")

    if output_dir is None:
        output_dir = os.path.join(out_root, "mosaic")
    os.makedirs(output_dir, exist_ok=True)

    # 1) Discover tile folders (named "<i>_<j>") and their pixel origins.
    tiles = []  # (i, j, folder_path)
    for name in sorted(os.listdir(out_root)):
        folder = os.path.join(out_root, name)
        parts = name.split("_")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and os.path.isdir(folder):
            tiles.append((int(parts[0]), int(parts[1]), folder))
    if not tiles:
        raise RuntimeError(f"No tile folders (named '<i>_<j>') found in {out_root}")
    tiles.sort(key=lambda t: (t[0], t[1]))

    # 2) Group output tiles by variable. The folder name is the authoritative
    #    (i, j), so the variable is whatever precedes the "_<i>_<j>.tif" suffix.
    var_tiles = {}  # var -> list of (i, j, path)
    for i, j, folder in tiles:
        suffix = f"_{i}_{j}.tif"
        for fname in os.listdir(folder):
            if fname.startswith(".") or not fname.endswith(suffix):
                continue
            var = fname[: -len(suffix)]
            if var:
                var_tiles.setdefault(var, []).append((i, j, os.path.join(folder, fname)))

    if variables is not None:
        missing = set(variables) - set(var_tiles)
        if missing:
            raise ValueError(
                f"Requested variables not found in {out_root}: {sorted(missing)}. "
                f"Available: {sorted(var_tiles)}"
            )
        var_names = [v for v in variables if v in var_tiles]
    else:
        var_names = sorted(var_tiles)

    if not var_names:
        raise RuntimeError(f"No variable GeoTIFFs found under tile folders in {out_root}")

    driver = gdal.GetDriverByName("GTiff")
    written = []

    for var in var_names:
        entries = sorted(var_tiles[var], key=lambda t: (t[0], t[1]))

        # --- reference tile: geometry, bands, projection, per-band Time ---
        ref_i, ref_j, ref_path = entries[0]
        ref_ds = gdal.Open(ref_path)
        if ref_ds is None:
            raise FileNotFoundError(f"Could not open {ref_path}")
        num_bands = ref_ds.RasterCount
        band_dtype = ref_ds.GetRasterBand(1).DataType
        projection = ref_ds.GetProjection()
        ref_gt = ref_ds.GetGeoTransform()
        px_w, px_h = ref_gt[1], ref_gt[5]
        if ref_gt[2] != 0 or ref_gt[4] != 0:
            raise ValueError(f"Rotated geotransform not supported ({var}, tile {ref_i}_{ref_j}).")
        band_times = [ref_ds.GetRasterBand(b + 1).GetMetadataItem("Time") for b in range(num_bands)]
        ref_ds = None

        # Global origin (source pixel 0,0), back-projected from the reference tile.
        origin_x = ref_gt[0] - ref_i * px_w
        origin_y = ref_gt[3] - ref_j * px_h

        # --- pass 1: validate every tile and size the mosaic ---
        placements = []  # (i, j, path, copy_w, copy_h)
        out_w = out_h = 0
        for i, j, path in entries:
            ds = gdal.Open(path)
            if ds is None:
                raise FileNotFoundError(f"Could not open {path}")
            gt = ds.GetGeoTransform()
            if ds.RasterCount != num_bands:
                raise ValueError(
                    f"Band count mismatch for {var}: tile {i}_{j} has {ds.RasterCount}, "
                    f"expected {num_bands}."
                )
            if ds.GetProjection() != projection:
                raise ValueError(f"Projection mismatch for {var} at tile {i}_{j}.")
            if gt[1] != px_w or gt[5] != px_h or gt[2] != 0 or gt[4] != 0:
                raise ValueError(f"Pixel size / rotation mismatch for {var} at tile {i}_{j}.")
            copy_w = min(tile_size, ds.RasterXSize)  # drop right overlap fringe
            copy_h = min(tile_size, ds.RasterYSize)  # drop bottom overlap fringe
            ds = None
            placements.append((i, j, path, copy_w, copy_h))
            out_w = max(out_w, i + copy_w)
            out_h = max(out_h, j + copy_h)

        creation_options = []
        if compress:
            creation_options = [f"COMPRESS={compress}", "TILED=YES", "BIGTIFF=IF_SAFER"]
            # Floating-point predictor for float rasters, horizontal for integers.
            if band_dtype in (gdal.GDT_Float32, gdal.GDT_Float64):
                creation_options.append("PREDICTOR=3")
            else:
                creation_options.append("PREDICTOR=2")

        # Only pre-fill with NaN when the tiles do not perfectly tile the mosaic
        # (partial tile_keys runs); a complete grid overwrites every pixel.
        covered = sum(cw * ch for (_, _, _, cw, ch) in placements)
        complete = covered == out_w * out_h

        if split_bands:
            # --- one single-band GeoTIFF per band (per hour); no multi-band file ---
            band_paths = []
            band_out = []  # (source_band_index, dataset)
            for b in range(num_bands):
                bpath = os.path.join(output_dir, _band_filename(var, band_times[b], b))
                band_paths.append(bpath)
                if os.path.exists(bpath):
                    if not overwrite:
                        print(f"[mosaic] {os.path.basename(bpath)}: exists, skipping (overwrite=False)")
                        continue
                    driver.Delete(bpath)
                ods = driver.Create(bpath, out_w, out_h, 1, band_dtype, creation_options)
                ods.SetGeoTransform((origin_x, px_w, 0.0, origin_y, 0.0, px_h))
                ods.SetProjection(projection)
                ob = ods.GetRasterBand(1)
                if not complete:
                    ob.Fill(NAN)
                if band_times[b] is not None:
                    ob.SetMetadata({"Time": band_times[b]})
                band_out.append((b, ods))
            # place each tile's core once, fanning its bands out to the per-hour files
            for i, j, path, copy_w, copy_h in placements:
                src = gdal.Open(path)
                for b, ods in band_out:
                    arr = src.GetRasterBand(b + 1).ReadAsArray(0, 0, copy_w, copy_h)
                    ods.GetRasterBand(1).WriteArray(arr, xoff=i, yoff=j)
                src = None
            for _, ods in band_out:
                ods.FlushCache()
            written.extend(band_paths)
            print(f"[mosaic] {var}: {out_w}x{out_h}, {len(band_out)} per-hour file(s)")
            continue

        # --- one multi-band GeoTIFF per variable ---
        out_path = os.path.join(output_dir, f"{var}.tif")
        if os.path.exists(out_path):
            if not overwrite:
                print(f"[mosaic] {var}: exists, skipping (overwrite=False)")
                written.append(out_path)
                continue
            driver.Delete(out_path)

        out_ds = driver.Create(out_path, out_w, out_h, num_bands, band_dtype, creation_options)
        out_ds.SetGeoTransform((origin_x, px_w, 0.0, origin_y, 0.0, px_h))
        out_ds.SetProjection(projection)
        for b in range(num_bands):
            ob = out_ds.GetRasterBand(b + 1)
            if not complete:
                ob.Fill(NAN)
            if band_times[b] is not None:
                ob.SetMetadata({"Time": band_times[b]})

        # place each tile's core at its pixel offset
        for i, j, path, copy_w, copy_h in placements:
            ds = gdal.Open(path)
            for b in range(num_bands):
                arr = ds.GetRasterBand(b + 1).ReadAsArray(0, 0, copy_w, copy_h)
                out_ds.GetRasterBand(b + 1).WriteArray(arr, xoff=i, yoff=j)
            ds = None

        out_ds.FlushCache()
        out_ds = None
        written.append(out_path)
        print(f"[mosaic] {var}: {out_w}x{out_h}, {num_bands} band(s) -> {out_path}")

    print(f"[mosaic] wrote {len(written)} mosaic(s) to {output_dir}")
    return output_dir
