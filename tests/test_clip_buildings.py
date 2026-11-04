"""
Tests for the clip_buildings option.

With clip_buildings=True (the default) building footprints (Building DSM - DEM
>= 2 m) are NaN in the UTCI and Shadow outputs; with False the rooftop values
are kept. Both outputs record the setting in their CLIP_BUILDINGS metadata item,
and run_utci_tiles(overwrite=False) recomputes tiles written with the other
setting instead of skipping them.
"""
import zipfile

import numpy as np
import pytest
from osgeo import gdal, osr

import solweig_gpu.utci_process as utci_process
from solweig_gpu import run_utci_tiles
from solweig_gpu.utci_process import CLIP_TAG, TREES_TAG, clip_state, tree_tag

gdal.UseExceptions()

DATE = "2024-05-29"
KEY = "0_0"
# Rows of the Mumbai 8000_16000 metfile (DOY 150; 09:30, 14:30, 17:30 local time).
MET = """iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd uhii
2024 150 9 30 -999.00 -999.00 -999.00 -999.00 -999.00 5.48295 67.03 31.13 100.47 0.00 430.97 -999.00 -999.00 -999.00 -999.00 -999.00 -999.00 -999.00 -999.00 -999.00 0.00
2024 150 14 30 -999.00 -999.00 -999.00 -999.00 -999.00 6.60346 59.67 33.15 100.30 0.00 870.48 -999.00 -999.00 -999.00 -999.00 -999.00 -999.00 -999.00 -999.00 -999.00 0.00
2024 150 17 30 -999.00 -999.00 -999.00 -999.00 -999.00 5.03232 62.74 31.84 100.18 0.00 371.73 -999.00 -999.00 -999.00 -999.00 -999.00 -999.00 -999.00 -999.00 -999.00 0.00
"""
TIMES = ["2024-05-29T09:30:00", "2024-05-29T14:30:00", "2024-05-29T17:30:00"]

# 2 m pixels in UTM 43N (Mumbai), so the tile resolves to Asia/Kolkata.
PIXEL = 2.0
GEOTRANSFORM = (276923.4, PIXEL, 0.0, 2109921.7, 0.0, -PIXEL)
H = W = 24

LEGACY = "legacy"  # output written before the option existed: no tag
EMPTY = "empty"    # unreadable output, e.g. from a run killed mid-write


def _write_tif(path, arr, tag=None, trees=None):
    arr = np.asarray(arr, dtype=np.float32)
    ds = gdal.GetDriverByName("GTiff").Create(str(path), arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    ds.SetGeoTransform(GEOTRANSFORM)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32643)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(arr)
    if tag is not None:
        ds.SetMetadataItem(CLIP_TAG, tag)
    if trees is not None:
        ds.SetMetadataItem(TREES_TAG, trees)
    ds = None


def _read_output(path):
    ds = gdal.Open(str(path))
    arr = ds.ReadAsArray()
    tag = ds.GetMetadataItem(CLIP_TAG)
    times = [ds.GetRasterBand(b + 1).GetMetadataItem("Time") for b in range(ds.RasterCount)]
    ds = None
    return arr, tag, times


def _write_metfile(pre):
    (pre / "metfiles").mkdir()
    (pre / "metfiles" / f"metfile_{KEY}_{DATE}.txt").write_text(MET)


# --- clip_state ---------------------------------------------------------------

@pytest.mark.parametrize("var,tag,expected", [
    ("UTCI", "YES", True),
    ("UTCI", "NO", False),
    ("Shadow", "YES", True),
    ("Shadow", "NO", False),
    ("UTCI", None, True),     # untagged: UTCI was always clipped before the option
    ("Shadow", None, False),  # untagged: Shadow never was
])
def test_clip_state_reads_tag(tmp_path, var, tag, expected):
    path = tmp_path / f"{var}_{KEY}.tif"
    _write_tif(path, np.zeros((4, 4)), tag=tag)
    assert clip_state(str(path), var) is expected


@pytest.mark.parametrize("content", [b"", b"not a tiff"])
def test_clip_state_unreadable_file_is_none(tmp_path, content):
    path = tmp_path / f"UTCI_{KEY}.tif"
    path.write_bytes(content)
    assert clip_state(str(path), "UTCI") is None


# --- run_utci_tiles resume logic (compute_utci stubbed) -----------------------

@pytest.fixture
def compute_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(utci_process, "compute_utci", lambda *args, **kwargs: calls.append(kwargs))
    return calls


@pytest.mark.parametrize("existing,kwargs,expect_run", [
    pytest.param({}, {}, True, id="no-outputs"),
    pytest.param({"UTCI": "YES", "Shadow": "YES"}, {"save_shadow": True}, False, id="tags-match"),
    pytest.param({"UTCI": "YES", "Shadow": "YES"}, {"save_shadow": True, "clip_buildings": False}, True,
                 id="tags-differ"),
    pytest.param({"UTCI": LEGACY}, {}, False, id="legacy-utci-clip"),
    pytest.param({"UTCI": LEGACY}, {"clip_buildings": False}, True, id="legacy-utci-noclip"),
    pytest.param({"UTCI": LEGACY, "Shadow": LEGACY}, {"save_shadow": True}, True, id="legacy-shadow-clip"),
    pytest.param({"UTCI": LEGACY, "Shadow": LEGACY}, {"save_shadow": True, "clip_buildings": False}, True,
                 id="legacy-shadow-noclip"),
    pytest.param({"UTCI": "YES", "Shadow": "NO"}, {}, False, id="unrequested-shadow-ignored"),
    pytest.param({"UTCI": "YES"}, {"overwrite": True}, True, id="overwrite"),
    pytest.param({"UTCI": EMPTY}, {}, True, id="unreadable-utci"),
])
def test_resume_respects_clip_setting(tmp_path, compute_calls, existing, kwargs, expect_run):
    pre = tmp_path / "processed_inputs"
    for prefix in ("Building_DSM", "Trees", "DEM", "walls", "aspect"):
        (pre / prefix).mkdir(parents=True)
        (pre / prefix / f"{prefix}_{KEY}.tif").touch()
    _write_metfile(pre)

    out = tmp_path / "output_folder" / KEY
    out.mkdir(parents=True)
    for var, tag in existing.items():
        path = out / f"{var}_{KEY}.tif"
        if tag == EMPTY:
            path.touch()
        else:
            # Tree tag as written by a default run, so only the clip setting differs.
            _write_tif(path, np.zeros((4, 4)), tag=None if tag == LEGACY else tag, trees=tree_tag(None))

    run_utci_tiles(base_path=str(tmp_path), preprocess_dir=str(pre), selected_date_str=DATE,
                   save_tmrt=False, **kwargs)

    assert len(compute_calls) == int(expect_run)
    if expect_run:
        assert compute_calls[0]["clip_buildings"] is kwargs.get("clip_buildings", True)


# --- compute_utci on a synthetic tile ------------------------------------------

def _synthetic_preprocess(tmp_path, veg=1.0):
    """
    Flat 24x24 tile with one 10 m building and a precomputed SVF cache.

    The cache sees open sky; with veg < 1 that share of the sky is seen through
    tree canopy instead (svf*veg rasters and vegshadowmat).
    """
    from solweig_gpu.shadow import create_patches
    from solweig_gpu.walls_aspect import filter1Goodwin_as_aspect_v3, findwalls, walllimit

    dem = np.full((H, W), 10.0, dtype=np.float32)
    dsm = dem.copy()
    dsm[8:16, 8:16] += 10.0
    walls = findwalls(dsm, walllimit)
    rasters = {
        "Building_DSM": dsm,
        "DEM": dem,
        "Trees": np.zeros((H, W), np.float32),
        "walls": walls,
        "aspect": filter1Goodwin_as_aspect_v3(walls, 1 / PIXEL, dsm),
    }
    pre = tmp_path / "processed_inputs"
    for prefix, arr in rasters.items():
        (pre / prefix).mkdir(parents=True)
        _write_tif(pre / prefix / f"{prefix}_{KEY}.tif", arr)
    _write_metfile(pre)

    # compute_utci loads this cache instead of running svf_calculator (and torch.compile).
    svf_dir = pre / "SVF"
    members = tmp_path / "svf_members"
    svf_dir.mkdir()
    members.mkdir()
    with zipfile.ZipFile(svf_dir / f"svfs_{KEY}.zip", "w") as zf:
        for direction in ("", "E", "S", "W", "N"):
            for suffix, value in (("", 0.9), ("veg", veg), ("aveg", 1.0)):
                name = f"svf{direction}{suffix}.tif"
                _write_tif(members / name, np.full((H, W), value))
                zf.write(members / name, name)
    _write_tif(svf_dir / f"SkyViewFactor_{KEY}.tif", np.full((H, W), 0.9))
    n_patches = int(create_patches(2)[4].sum())
    ones = np.ones((H, W, n_patches), np.float32)
    np.savez(svf_dir / f"shadowmats_{KEY}.npz", shadowmat=ones, vegshadowmat=ones * np.float32(veg), vbshmat=ones)

    return pre, (dsm - dem) >= 2


def test_clip_buildings_masks_only_roofs(tmp_path, monkeypatch):
    def no_svf(*args, **kwargs):
        raise AssertionError("SVF cache was not used")
    monkeypatch.setattr(utci_process, "svf_calculator", no_svf)

    pre, roof = _synthetic_preprocess(tmp_path)
    common = dict(preprocess_dir=str(pre), selected_date_str=DATE, save_tmrt=False, save_shadow=True)
    run_utci_tiles(base_path=str(tmp_path / "clip"), clip_buildings=True, **common)
    run_utci_tiles(base_path=str(tmp_path / "noclip"), clip_buildings=False, **common)

    for var in ("UTCI", "Shadow"):
        on, on_tag, on_times = _read_output(tmp_path / "clip" / "output_folder" / KEY / f"{var}_{KEY}.tif")
        off, off_tag, off_times = _read_output(tmp_path / "noclip" / "output_folder" / KEY / f"{var}_{KEY}.tif")
        assert on.shape == off.shape == (len(TIMES), H, W)
        assert (on_tag, off_tag) == ("YES", "NO")
        assert on_times == off_times == TIMES
        assert np.isnan(on[:, roof]).all(), var
        assert np.isfinite(on[:, ~roof]).all(), var
        assert np.isfinite(off).all(), var  # rooftop values kept
        np.testing.assert_array_equal(on[:, ~roof], off[:, ~roof])

    # Same setting again: the tile is complete, so it must be skipped.
    def must_not_run(*args, **kwargs):
        raise AssertionError("tile should have been skipped")
    monkeypatch.setattr(utci_process, "compute_utci", must_not_run)
    run_utci_tiles(base_path=str(tmp_path / "clip"), clip_buildings=True, **common)
