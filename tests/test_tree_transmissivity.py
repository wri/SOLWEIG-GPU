"""
Tests for the tree_transmissivity_pct option.

By default (None) trees are leafless outside days 98-299, a temperate deciduous
season, so 50% of the direct sun gets through the canopy instead of 3%. A
percentage applies on every date, to the direct sun and the diffuse sky light.
UTCI and Shadow record the setting in their TREE_TRANSMISSIVITY_PCT metadata
item, and run_utci_tiles(overwrite=False) recomputes tiles written with another
value or before the item existed.
"""
import numpy as np
import pytest
from osgeo import gdal

import solweig_gpu.utci_process as utci_process
from solweig_gpu import run_utci_tiles
from solweig_gpu.utci_process import TREES_TAG, tree_state, tree_tag

from .test_clip_buildings import DATE, H, KEY, W, _synthetic_preprocess, _write_metfile, _write_tif

gdal.UseExceptions()


def _read(path):
    ds = gdal.Open(str(path))
    arr = ds.ReadAsArray()
    tag = ds.GetMetadataItem(TREES_TAG)
    ds = None
    return arr, tag


def _stub_inputs(tmp_path):
    """Empty input tiles and a metfile: enough for run_utci_tiles with compute_utci stubbed."""
    pre = tmp_path / "processed_inputs"
    for prefix in ("Building_DSM", "Trees", "DEM", "walls", "aspect"):
        (pre / prefix).mkdir(parents=True)
        (pre / prefix / f"{prefix}_{KEY}.tif").touch()
    _write_metfile(pre)
    return pre


@pytest.fixture
def compute_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(utci_process, "compute_utci", lambda *args, **kwargs: calls.append(kwargs))
    return calls


# --- tree_tag / tree_state -------------------------------------------------------

@pytest.mark.parametrize("pct,tag", [
    (None, "seasonal"),
    (3, "3"),
    (3.0, "3"),
    (12.5, "12.5"),
    (-0.0, "0"),
    (100, "100"),
])
def test_tree_tag(pct, tag):
    assert tree_tag(pct) == tag


@pytest.mark.parametrize("tag", ["3", "seasonal", None])
def test_tree_state_reads_tag(tmp_path, tag):
    path = tmp_path / f"UTCI_{KEY}.tif"
    _write_tif(path, np.zeros((4, 4)), trees=tag)
    assert tree_state(str(path)) == tag


@pytest.mark.parametrize("content", [b"", b"not a tiff"])
def test_tree_state_unreadable_file_is_none(tmp_path, content):
    path = tmp_path / f"UTCI_{KEY}.tif"
    path.write_bytes(content)
    assert tree_state(str(path)) is None


# --- run_utci_tiles (compute_utci stubbed) ----------------------------------------

@pytest.mark.parametrize("kwargs,expected", [
    pytest.param({}, None, id="default"),
    pytest.param({"tree_transmissivity_pct": 3}, 3.0, id="3"),
    pytest.param({"tree_transmissivity_pct": 12.5}, 12.5, id="12.5"),
])
def test_run_utci_tiles_forwards_the_percentage(tmp_path, compute_calls, kwargs, expected):
    run_utci_tiles(base_path=str(tmp_path), preprocess_dir=str(_stub_inputs(tmp_path)), selected_date_str=DATE,
                   save_tmrt=False, **kwargs)

    assert len(compute_calls) == 1
    assert compute_calls[0]["tree_transmissivity_pct"] == expected


@pytest.mark.parametrize("pct", [-1, 100.5, float("nan"), True])
def test_percentage_must_be_from_0_to_100(tmp_path, compute_calls, pct):
    with pytest.raises(ValueError, match="tree_transmissivity_pct"):
        run_utci_tiles(base_path=str(tmp_path), preprocess_dir=str(_stub_inputs(tmp_path)),
                       selected_date_str=DATE, save_tmrt=False, tree_transmissivity_pct=pct)
    assert compute_calls == []


@pytest.mark.parametrize("written,kwargs,expect_run", [
    pytest.param({"UTCI": "3"}, {"tree_transmissivity_pct": 3}, False, id="tags-match"),
    pytest.param({"UTCI": "3"}, {"tree_transmissivity_pct": 10}, True, id="tags-differ"),
    pytest.param({"UTCI": "seasonal"}, {}, False, id="default-matches"),
    pytest.param({"UTCI": "seasonal"}, {"tree_transmissivity_pct": 3}, True, id="default-then-percent"),
    pytest.param({"UTCI": None}, {"tree_transmissivity_pct": 3}, True, id="untagged"),
    pytest.param({"UTCI": None}, {}, True, id="untagged-default"),
    pytest.param({"UTCI": "3", "Shadow": "3"}, {"tree_transmissivity_pct": 3, "save_shadow": True}, False,
                 id="shadow-matches"),
    pytest.param({"UTCI": "3", "Shadow": "10"}, {"tree_transmissivity_pct": 3, "save_shadow": True}, True,
                 id="stale-shadow"),
    pytest.param({"UTCI": "3", "Shadow": "10"}, {"tree_transmissivity_pct": 3}, False,
                 id="unrequested-shadow-ignored"),
])
def test_resume_respects_tree_setting(tmp_path, compute_calls, written, kwargs, expect_run):
    pre = _stub_inputs(tmp_path)
    out = tmp_path / "output_folder" / KEY
    out.mkdir(parents=True)
    for var, trees in written.items():
        # Clip tag as written by a default run (no albedo tag also means the defaults),
        # so only the tree setting differs.
        _write_tif(out / f"{var}_{KEY}.tif", np.zeros((4, 4)), tag="YES", trees=trees)

    run_utci_tiles(base_path=str(tmp_path), preprocess_dir=str(pre), selected_date_str=DATE, save_tmrt=False,
                   **kwargs)

    assert len(compute_calls) == int(expect_run)
    if expect_run:
        assert compute_calls[0]["tree_transmissivity_pct"] == kwargs.get("tree_transmissivity_pct")


# --- compute_utci on a synthetic tile ----------------------------------------------

def test_percentage_sets_the_direct_sun_through_trees(tmp_path, monkeypatch):
    pre, _ = _synthetic_preprocess(tmp_path)
    trees = np.zeros((H, W), np.float32)
    trees[2:6, 2:6] = 8.0  # an 8 m tree clear of the building
    _write_tif(pre / "Trees" / f"Trees_{KEY}.tif", trees)
    # Move the leaf season so that the metfile's day (DOY 150) falls outside it.
    monkeypatch.setattr(utci_process, "firstdayleaf", 200)

    shadow, tags = {}, {}
    for name, pct in (("leafless", None), ("3", 3), ("25", 25)):
        out = tmp_path / name / "output_folder" / KEY
        run_utci_tiles(base_path=str(tmp_path / name), preprocess_dir=str(pre), selected_date_str=DATE,
                       save_tmrt=False, save_shadow=True, tree_transmissivity_pct=pct)
        shadow[name], shadow_tag = _read(out / f"Shadow_{KEY}.tif")
        _, utci_tag = _read(out / f"UTCI_{KEY}.tif")
        tags[name] = (utci_tag, shadow_tag)

    tree_shade = np.isclose(shadow["leafless"], 0.5)
    assert tree_shade.any()
    np.testing.assert_allclose(shadow["3"][tree_shade], 0.03, atol=1e-6)
    np.testing.assert_allclose(shadow["25"][tree_shade], 0.25, atol=1e-6)
    for name in ("3", "25"):
        np.testing.assert_array_equal(shadow[name][~tree_shade], shadow["leafless"][~tree_shade])
    assert tags == {"leafless": ("seasonal", "seasonal"), "3": ("3", "3"), "25": ("25", "25")}


def test_percentage_sets_the_diffuse_light_through_trees(tmp_path):
    # Half of the sky is seen through canopy, but no tree casts a shadow, so only
    # the diffuse sky light depends on the setting.
    pre, _ = _synthetic_preprocess(tmp_path, veg=0.5)

    kdown, shadow = {}, {}
    for name, pct in (("default", None), ("3", 3), ("50", 50)):
        out = tmp_path / name / "output_folder" / KEY
        run_utci_tiles(base_path=str(tmp_path / name), preprocess_dir=str(pre), selected_date_str=DATE,
                       save_tmrt=False, save_kdown=True, save_shadow=True, tree_transmissivity_pct=pct)
        kdown[name], _ = _read(out / f"Kdown_{KEY}.tif")
        shadow[name], _ = _read(out / f"Shadow_{KEY}.tif")

    for name in ("3", "50"):
        np.testing.assert_array_equal(shadow[name], shadow["default"])
    # The metfile's day (DOY 150) is in the leaf season, where the default lets 3% through.
    np.testing.assert_array_equal(kdown["3"], kdown["default"])
    assert (kdown["50"] > kdown["3"]).all()
