"""
Tests for land cover inputs: OpenUrban conversion, the SOLWEIG code check in
preprocess, stale Landcover/ folders, and the recoded grid reaching the solver.
"""
import inspect

import numpy as np
import pytest
from osgeo import gdal

import solweig_gpu.utci_process as utci_process
from solweig_gpu import run_utci_tiles
from solweig_gpu.landcover import check_landcover_codes, openurban_to_solweig

from .test_albedo import _inputs, _preprocess, _write_raster
from .test_clip_buildings import DATE, H, KEY, W, _synthetic_preprocess, _write_tif

gdal.UseExceptions()

# CIF's city_metrix.layers.open_urban.reclass_map (applied by CTCM), spelled out.
CIF_RECLASS = {110: 5, 120: 1, 130: 6, 200: 5, 300: 7, 400: 1, 500: 1,
               600: 2, 601: 2, 602: 2, 610: 2, 611: 2, 612: 2, 620: 2, 621: 2, 622: 2}


def _read(path):
    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    info = (band.ReadAsArray(), gdal.GetDataTypeName(band.DataType), band.GetNoDataValue(),
            ds.GetGeoTransform(), ds.GetProjection())
    ds = None
    return info


def test_openurban_codes_map_as_in_cif(tmp_path):
    # Every OpenUrban code, then NaN, an unknown code and one past the table.
    codes = list(CIF_RECLASS) + [np.nan, 115, 999]
    arr = np.full(H * W, 500, np.float32)
    arr[:len(codes)] = codes
    src = _write_raster(tmp_path / "OpenUrban.tif", arr.reshape(H, W))

    counts = openurban_to_solweig(str(src), str(tmp_path / "Landcover.tif"))

    out, dtype, nodata, geotransform, projection = _read(tmp_path / "Landcover.tif")
    _, _, _, src_geotransform, src_projection = _read(src)
    assert out.ravel()[:len(codes)].tolist() == list(CIF_RECLASS.values()) + [0, 0, 0]
    assert (out.ravel()[len(codes):] == 1).all()  # roads -> paved
    assert (dtype, nodata) == ("Byte", 0)
    assert (geotransform, projection) == (src_geotransform, src_projection)
    assert counts == {int(c): int((out == c).sum()) for c in np.unique(out)}


def test_check_landcover_codes(tmp_path):
    solweig = _write_raster(tmp_path / "solweig.tif", np.tile([1, 2, 5, 6, 7, 3], (H, W // 6)))
    assert check_landcover_codes(str(solweig)) == 0.0

    openurban = _write_raster(tmp_path / "openurban.tif", np.full((H, W), 110.0))
    with pytest.raises(ValueError, match="python -m solweig_gpu.landcover"):
        check_landcover_codes(str(openurban))

    other = _write_raster(tmp_path / "other.tif", np.full((H, W), 50.0))
    with pytest.raises(ValueError, match="Reclassify"):
        check_landcover_codes(str(other))


def test_preprocess_rejects_openurban_codes(tmp_path, capsys):
    base = _inputs(tmp_path)
    _write_raster(base / "OpenUrban.tif", np.full((H, W), 500.0))
    with pytest.raises(SystemExit):
        _preprocess(tmp_path, base, landcover_filename="OpenUrban.tif")
    assert "OpenUrban" in capsys.readouterr().out


def test_preprocess_drops_stale_land_cover(tmp_path):
    base = _inputs(tmp_path)
    _write_raster(base / "Landcover.tif", np.full((H, W), 5.0))
    pre = _preprocess(tmp_path, base, landcover_filename="Landcover.tif")
    assert (pre / "Landcover" / f"Landcover_{KEY}.tif").exists()

    # run_utci_tiles uses Landcover/ whenever it exists, so a run without land
    # cover must not leave the earlier tiles behind.
    _preprocess(tmp_path, base)
    assert not (pre / "Landcover").exists()


def test_solver_gets_the_recoded_land_cover(tmp_path, monkeypatch):
    pre, _ = _synthetic_preprocess(tmp_path)
    landcover = np.full((H, W), 1.0, np.float32)
    landcover[0:4] = 3.0    # evergreen trees
    landcover[4:8] = 4.0    # deciduous trees
    landcover[20:24] = 9.0  # not a SOLWEIG code
    (pre / "Landcover").mkdir()
    _write_tif(pre / "Landcover" / f"Landcover_{KEY}.tif", landcover)

    solver = utci_process.Solweig_2022a_calc
    signature = inspect.signature(solver)
    seen = []

    def spy(*args, **kwargs):
        seen.append(signature.bind(*args, **kwargs).arguments["lc_grid"].cpu().numpy().copy())
        return solver(*args, **kwargs)

    monkeypatch.setattr(utci_process, "Solweig_2022a_calc", spy)
    run_utci_tiles(base_path=str(tmp_path), preprocess_dir=str(pre), selected_date_str=DATE, save_tmrt=False)

    # Trees -> grass and unknown codes -> bare soil, as for the surface parameters;
    # the raw codes would make the solver treat code 3 as water on the GPU.
    assert seen
    for grid in seen:
        np.testing.assert_array_equal(grid[0:8], 5)
        np.testing.assert_array_equal(grid[8:20], 1)
        np.testing.assert_array_equal(grid[20:24], 6)
