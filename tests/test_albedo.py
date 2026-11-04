"""
Tests for the albedo inputs.

An albedo raster (preprocess albedo_filename -> Albedo tiles) replaces the
ground albedo; albedo_walls / albedo_ground replace the fixed 0.2 / 0.15; a
baseline albedo raster sets the cool-roof air-temperature cut. The UTCI output
records these settings in its ALBEDO metadata item, and run_utci_tiles
(overwrite=False) recomputes tiles written with other settings.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

import solweig_gpu.utci_process as utci_process
from solweig_gpu import preprocess, run_utci_tiles
from solweig_gpu.albedo import (AIR_TEMP_CUT_FILE, ALBEDO_TAG, KRAYENHOFF_COEFF, LEGACY_ALBEDO_TAG,
                                air_temp_cut_factor, albedo_tag, check_albedo_range, read_air_temp_cut,
                                write_air_temp_cut)
from solweig_gpu.utci_process import tree_tag

from .test_clip_buildings import (DATE, GEOTRANSFORM, H, KEY, MET, W, _read_output, _synthetic_preprocess,
                                  _write_metfile, _write_tif)

gdal.UseExceptions()

OUTPUTS = ("UTCI", "TMRT", "Kup", "Kdown", "Shadow", "Ta")
# The MET rows are at 09:30, 14:30 and 17:30 local time.
MET_HOURS = [9.5, 14.5, 17.5]


def _write_raster(path, arr, geotransform=GEOTRANSFORM, nodata=None):
    arr = np.asarray(arr, dtype=np.float32)
    ds = gdal.GetDriverByName("GTiff").Create(str(path), arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32643)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(arr)
    ds = None
    return path


def _write_albedo(pre, arr, nodata=None):
    (pre / "Albedo").mkdir(exist_ok=True)
    return _write_raster(pre / "Albedo" / f"Albedo_{KEY}.tif", arr, nodata=nodata)


def _albedo_tag_of(path):
    ds = gdal.Open(str(path))
    value = ds.GetMetadataItem(ALBEDO_TAG)
    ds = None
    return value


def _run(base, pre, **kwargs):
    """Run the synthetic tile; return {output: array} and the UTCI ALBEDO tag."""
    run_utci_tiles(base_path=str(base), preprocess_dir=str(pre), selected_date_str=DATE, save_tmrt=True,
                   save_kup=True, save_kdown=True, save_shadow=True, save_ta=True, clip_buildings=False,
                   **kwargs)
    out = Path(base) / "output_folder" / KEY
    arrays = {var: _read_output(out / f"{var}_{KEY}.tif")[0] for var in OUTPUTS}
    return arrays, _albedo_tag_of(out / f"UTCI_{KEY}.tif")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Synthetic preprocess dir (no albedo) and its default-albedo outputs."""
    root = tmp_path_factory.mktemp("albedo")
    pre, _ = _synthetic_preprocess(root)
    default, tag = _run(root / "default", pre)
    return pre, default, tag


def _copy(pre, tmp_path):
    copy = tmp_path / "processed_inputs"
    shutil.copytree(pre, copy)
    return copy


# --- albedo raster on the synthetic tile -----------------------------------------

def test_default_run_is_tagged_like_legacy_outputs(synthetic):
    _, _, tag = synthetic
    assert tag == LEGACY_ALBEDO_TAG == "raster=none;ground=0.15;walls=0.2;cut=0"


def test_constant_albedo_raster_matches_default(synthetic, tmp_path):
    pre, default, _ = synthetic
    pre = _copy(pre, tmp_path)
    tile = _write_albedo(pre, np.full((H, W), 0.15))
    arrays, tag = _run(tmp_path, pre)
    for var in OUTPUTS:
        np.testing.assert_array_equal(arrays[var], default[var], err_msg=var)
    assert tag == albedo_tag(str(tile), 0.15, 0.2, 0.0) != LEGACY_ALBEDO_TAG


def test_missing_albedo_pixels_keep_the_default(synthetic, tmp_path):
    pre, default, _ = synthetic
    pre = _copy(pre, tmp_path)
    albedo = np.full((H, W), 0.15, np.float32)
    albedo[:4, :4] = np.nan
    albedo[20:, 20:] = -9999.0
    _write_albedo(pre, albedo, nodata=-9999.0)
    arrays, _ = _run(tmp_path, pre)
    for var in OUTPUTS:
        np.testing.assert_array_equal(arrays[var], default[var], err_msg=var)


def test_brighter_ground_reflects_more_but_casts_the_same_shade(synthetic, tmp_path):
    pre, default, _ = synthetic
    pre = _copy(pre, tmp_path)
    _write_albedo(pre, np.full((H, W), 0.3))
    arrays, _ = _run(tmp_path, pre)
    assert (arrays["Kup"] > default["Kup"]).all()
    assert (arrays["TMRT"] > default["TMRT"]).all()
    for var in ("Kdown", "Shadow", "Ta"):
        np.testing.assert_array_equal(arrays[var], default[var], err_msg=var)


def test_percent_albedo_raster_is_rejected(synthetic, tmp_path):
    pre, _, _ = synthetic
    pre = _copy(pre, tmp_path)
    _write_albedo(pre, np.full((H, W), 15.0))
    with pytest.raises(ValueError, match="percent"):
        _run(tmp_path, pre)


# --- wall / ground albedo settings ---------------------------------------------

def test_wall_albedo(synthetic, tmp_path):
    pre, default, _ = synthetic
    same, same_tag = _run(tmp_path / "same", pre, albedo_walls=0.2, albedo_ground=0.15)
    for var in OUTPUTS:
        np.testing.assert_array_equal(same[var], default[var], err_msg=var)
    assert same_tag == LEGACY_ALBEDO_TAG

    bright, tag = _run(tmp_path / "bright", pre, albedo_walls=0.5)
    assert (bright["Kdown"] > default["Kdown"]).all()
    assert tag == albedo_tag(None, 0.15, 0.5, 0.0)


@pytest.mark.parametrize("kwargs", [{"albedo_walls": 1.5}, {"albedo_ground": -0.1}])
def test_albedo_settings_must_be_fractions(tmp_path, kwargs):
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_utci_tiles(base_path=str(tmp_path), preprocess_dir=str(tmp_path), selected_date_str=DATE, **kwargs)


# --- cool-roof air-temperature cut -----------------------------------------------

@pytest.mark.parametrize("hour,factor", [
    (9.5, 0.0),
    (11.99, 0.0),
    (12.0, 1.0),
    (14.5, 1.0 - 2.5 / 3 * (1.0 - 0.935315)),
    (15.0, 0.935315),
    (17.5, 0.935315 - 2.5 / 3 * (0.935315 - 0.646853)),
    (18.0, 0.646853),
    (18.01, 0.0),
])
def test_air_temp_cut_factor(hour, factor):
    assert float(air_temp_cut_factor(hour)) == pytest.approx(factor)


def test_air_temp_cut_lowers_the_forcing_air_temperature(synthetic, tmp_path):
    pre, default, _ = synthetic
    arrays, tag = _run(tmp_path, pre, air_temp_cut=0.52)
    for band, factor in enumerate(air_temp_cut_factor(MET_HOURS)):
        np.testing.assert_allclose(arrays["Ta"][band], default["Ta"][band] - 0.52 * factor, atol=1e-5)
    # 09:30 is outside 12:00-18:00, so nothing changes there.
    np.testing.assert_array_equal(arrays["UTCI"][0], default["UTCI"][0])
    assert (arrays["UTCI"][1:] < default["UTCI"][1:]).all()
    assert tag == albedo_tag(None, 0.15, 0.2, 0.52)


# --- run_utci_tiles resume logic (compute_utci stubbed) ----------------------------

@pytest.fixture
def compute_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(utci_process, "compute_utci", lambda *args, **kwargs: calls.append(kwargs))
    return calls


OTHER_RASTER = "raster=0000000000000000;ground=0.15;walls=0.2;cut=0"


@pytest.mark.parametrize("written,with_tile,sidecar,kwargs,expect_run", [
    pytest.param(None, False, None, {}, False, id="legacy-defaults"),
    pytest.param(None, True, None, {}, True, id="legacy-then-raster"),
    pytest.param("match", True, None, {}, False, id="raster-matches"),
    pytest.param(OTHER_RASTER, True, None, {}, True, id="raster-changed"),
    pytest.param(None, False, None, {"albedo_walls": 0.3}, True, id="walls-changed"),
    pytest.param(None, False, None, {"albedo_ground": 0.2}, True, id="ground-changed"),
    pytest.param(None, False, None, {"air_temp_cut": 0.5}, True, id="cut-given"),
    pytest.param("match", True, 0.52, {}, False, id="sidecar-cut-matches"),
    pytest.param(None, True, 0.52, {}, True, id="sidecar-cut-new"),
    pytest.param("match", True, 0.52, {"air_temp_cut": 0.0}, True, id="sidecar-cut-overridden"),
])
def test_resume_respects_albedo_settings(tmp_path, compute_calls, written, with_tile, sidecar, kwargs,
                                         expect_run):
    pre = tmp_path / "processed_inputs"
    for prefix in ("Building_DSM", "Trees", "DEM", "walls", "aspect"):
        (pre / prefix).mkdir(parents=True)
        (pre / prefix / f"{prefix}_{KEY}.tif").touch()
    _write_metfile(pre)
    tile = str(_write_albedo(pre, np.full((H, W), 0.2))) if with_tile else None
    if sidecar is not None:
        write_air_temp_cut(str(pre / "Albedo"), sidecar / KRAYENHOFF_COEFF, "scenario.tif", "baseline.tif")

    # The existing UTCI: clip and tree tags matching the defaults, albedo tag as parametrized
    # ("match": what the previous run with default arguments wrote; None: untagged).
    out = tmp_path / "output_folder" / KEY
    out.mkdir(parents=True)
    utci = out / f"UTCI_{KEY}.tif"
    _write_tif(utci, np.zeros((4, 4)), tag="YES", trees=tree_tag(None))
    if written == "match":
        written = albedo_tag(tile, 0.15, 0.2, sidecar or 0.0)
    if written is not None:
        ds = gdal.Open(str(utci), gdal.GA_Update)
        ds.SetMetadataItem(ALBEDO_TAG, written)
        ds = None

    run_utci_tiles(base_path=str(tmp_path), preprocess_dir=str(pre), selected_date_str=DATE, save_tmrt=False,
                   **kwargs)

    assert len(compute_calls) == int(expect_run)
    if expect_run:
        call = compute_calls[0]
        assert call["albedo_path"] == tile
        assert call["albedo_walls"] == kwargs.get("albedo_walls", 0.2)
        assert call["albedo_ground"] == kwargs.get("albedo_ground", 0.15)
        assert call["air_temp_cut"] == pytest.approx(kwargs.get("air_temp_cut", sidecar or 0.0))


# --- preprocess ------------------------------------------------------------------

def _inputs(tmp_path):
    """Building DSM (one 10 m building), DEM and Trees on the synthetic grid, plus a metfile."""
    base = tmp_path / "inputs"
    base.mkdir()
    dem = np.full((H, W), 10.0)
    dsm = dem.copy()
    dsm[8:16, 8:16] += 10.0
    _write_raster(base / "Building_DSM.tif", dsm)
    _write_raster(base / "DEM.tif", dem)
    _write_raster(base / "Trees.tif", np.zeros((H, W)))
    (tmp_path / "met.txt").write_text(MET)
    return base


def _preprocess(tmp_path, base, **kwargs):
    return Path(preprocess(base_path=str(base), selected_date_str=DATE, use_own_met=True,
                           own_met_file=str(tmp_path / "met.txt"), **kwargs))


def test_preprocess_resamples_albedo_onto_the_dsm_grid(tmp_path):
    base = _inputs(tmp_path)
    coarse = np.linspace(0.1, 0.6, 64, dtype=np.float32).reshape(8, 8)
    # 10 m pixels, extending past the 48 m x 48 m domain on every side.
    _write_raster(base / "Albedo.tif", coarse, geotransform=(276910.0, 10.0, 0.0, 2109940.0, 0.0, -10.0))
    pre = _preprocess(tmp_path, base, albedo_filename="Albedo.tif")

    ds = gdal.Open(str(pre / "Albedo" / f"Albedo_{KEY}.tif"))
    tile = ds.ReadAsArray()
    geotransform = ds.GetGeoTransform()
    ds = None
    assert tile.shape == (H, W)
    np.testing.assert_allclose(geotransform, GEOTRANSFORM)
    assert np.isfinite(tile).all()
    assert coarse.min() - 1e-6 <= tile.min() and tile.max() <= coarse.max() + 1e-6
    assert (pre / "albedo_aligned.vrt").exists()


def test_preprocess_records_the_air_temp_cut_and_drops_stale_albedo(tmp_path):
    base = _inputs(tmp_path)
    baseline = np.full((H, W), 0.15, np.float32)
    cool_roofs = baseline.copy()
    cool_roofs[8:16, 8:16] = 0.62  # the building's roof
    _write_raster(base / "Baseline.tif", baseline)
    _write_raster(base / "CoolRoofs.tif", cool_roofs)
    pre = _preprocess(tmp_path, base, albedo_filename="CoolRoofs.tif", baseline_albedo_filename="Baseline.tif")

    delta = (0.62 - 0.15) * 64 / (H * W)
    record = json.loads((pre / "Albedo" / AIR_TEMP_CUT_FILE).read_text())
    assert record["delta_albedo"] == pytest.approx(delta, rel=1e-6)
    assert record["air_temp_cut_noon_c"] == pytest.approx(KRAYENHOFF_COEFF * delta, rel=1e-6)
    assert read_air_temp_cut(str(pre / "Albedo")) == pytest.approx(KRAYENHOFF_COEFF * delta, rel=1e-6)
    assert (pre / "Albedo" / f"Albedo_{KEY}.tif").exists()
    assert not (pre / "albedo_aligned.vrt").exists()  # already on the DSM grid

    # A later run without albedo must not reuse these tiles or the cut.
    _preprocess(tmp_path, base)
    assert not (pre / "Albedo").exists()


def test_preprocess_rejects_percent_albedo(tmp_path, capsys):
    base = _inputs(tmp_path)
    _write_raster(base / "Albedo.tif", np.full((H, W), 15.0))
    with pytest.raises(SystemExit):
        _preprocess(tmp_path, base, albedo_filename="Albedo.tif")
    assert "percent" in capsys.readouterr().out


def test_preprocess_baseline_needs_an_albedo_raster(tmp_path, capsys):
    base = _inputs(tmp_path)
    _write_raster(base / "Baseline.tif", np.full((H, W), 0.15))
    with pytest.raises(SystemExit):
        _preprocess(tmp_path, base, baseline_albedo_filename="Baseline.tif")
    assert "requires albedo_filename" in capsys.readouterr().out


def test_all_nan_albedo_raster_is_rejected(tmp_path):
    path = _write_raster(tmp_path / "nan.tif", np.full((4, 4), np.nan))
    with pytest.raises(ValueError, match="no valid pixels"):
        check_albedo_range(str(path))
