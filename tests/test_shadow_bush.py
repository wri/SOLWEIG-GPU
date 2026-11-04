"""
Regression tests for the per-patch shadow march (``solweig_gpu.shadow.shadow``).

The bush branch of ``shadow()`` (executed only for tiles where ``bush.max() > 0``)
used to subtract 1 from its slice start indices, a leftover from the 1-based
MATLAB original. Any march step that shifts along an axis then wrapped a start
index of 0 to -1 and raised::

    RuntimeError: The expanded size of the tensor (H) must match the existing
    size (0) at non-singleton dimension 0.

These tests drive that branch from every azimuth quadrant on a tiny synthetic
tile, pin the reference semantics of a sub-threshold bush, and check that the
per-patch path agrees exactly with the batched path on a bush-free tile.
"""
import pytest
import torch

from solweig_gpu.shadow import shadow, shadow_batch

# shadow() places the azimuth on this device via ensure_tensor(), so every input
# has to live on it too.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE = 1.0  # 1 m pixels


def _make_tile(bush_h, H=24, W=18):
    """Flat ground with a building block, a tree block and (if ``bush_h > 0``) a
    3x3 bush block (canopy with no trunk zone).

    Built with the same recipe as ``svf_calculator``: ``vegdem``/``vegdem2`` are
    absolute elevations and 0 where there is no vegetation. The grid is
    deliberately non-square so row/column mix-ups cannot cancel out.
    """
    a = torch.zeros((H, W), device=DEVICE)
    a[4:8, 4:8] = 10.0                      # building
    trees = torch.zeros((H, W), device=DEVICE)
    trees[12:16, 6:10] = 6.0                # trees: canopy 6 m, trunk zone 1.5 m
    bush = torch.zeros((H, W), device=DEVICE)
    if bush_h > 0:
        trees[16:19, 12:15] = bush_h        # bush: canopy bush_h m, no trunk zone
        bush[16:19, 12:15] = bush_h
    vegdem = trees + a
    vegdem[vegdem == a] = 0.0
    vegdem2 = trees * 0.25 + a
    vegdem2[vegdem2 == a] = 0.0
    vegdem2[bush > 0] = 0.0                 # no trunk zone under a bush
    amaxvalue = torch.maximum(a.max(), vegdem.max()) - a.min()
    return amaxvalue, a, vegdem, vegdem2, bush


def _t(x):
    return torch.tensor(float(x), device=DEVICE)


def _run(inputs, azimuth, altitude, bush=None):
    amaxvalue, a, vegdem, vegdem2, tile_bush = inputs
    if bush is None:
        bush = tile_bush
    return shadow(amaxvalue, a, vegdem, vegdem2, bush, _t(azimuth), _t(altitude), SCALE)


def _assert_binary(t, shape):
    assert tuple(t.shape) == shape
    assert torch.isfinite(t).all()
    assert torch.all((t == 0.0) | (t == 1.0))


@pytest.mark.parametrize("azimuth", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
def test_bush_branch_runs_in_every_quadrant(azimuth):
    # bush_h > 1 so bushplant is set and the full bush logic runs. The 3x3 bush
    # block guarantees that at march step 1 a shifted bush pixel sits above the
    # ground in every direction, so torch.max(fabovea * bush) > 0 and the bush
    # branch executes for every azimuth. Azimuth 0 is remapped to 1e-12 inside
    # shadow(), which is exactly the dx=-1, dy=0 case that used to crash.
    inputs = _make_tile(bush_h=3.0)
    H, W = inputs[1].shape
    sh, vegsh, vbshvegsh = _run(inputs, azimuth, altitude=6.0)
    for out in (sh, vegsh, vbshvegsh):
        _assert_binary(out, (H, W))


@pytest.mark.parametrize("azimuth", [0.0, 90.0, 225.0])
def test_sub_threshold_bush_is_a_no_op(azimuth):
    # bush.max() > 0 routes the tile through the bush branch, but bushplant is
    # (bush > 1) and therefore empty: g is zeroed every step and the final
    # `g - bush` is <= 0, so the outputs must equal the bush-free result exactly.
    inputs = _make_tile(bush_h=0.75)
    zeros = torch.zeros_like(inputs[1])
    with_bush = _run(inputs, azimuth, altitude=6.0)
    without_bush = _run(inputs, azimuth, altitude=6.0, bush=zeros)
    for got, want in zip(with_bush, without_bush):
        assert torch.equal(got, want)


def test_bush_free_shadow_matches_shadow_batch():
    # The batched march is only used for bush-free tiles and must reproduce the
    # per-patch march bit for bit (this is what svf_calculator relies on).
    inputs = _make_tile(bush_h=0.0)
    amaxvalue, a, vegdem, vegdem2, bush = inputs
    assert bush.max() == 0.0
    azimuths = torch.tensor([0.0, 30.0, 90.0, 135.0, 180.0, 250.0, 270.0, 330.0], device=DEVICE)
    altitude = _t(30.0)
    sh_b, vegsh_b, vbsh_b = shadow_batch(amaxvalue, a, vegdem, vegdem2, azimuths, altitude, SCALE)
    for p, az in enumerate(azimuths.tolist()):
        sh, vegsh, vbshvegsh = shadow(amaxvalue, a, vegdem, vegdem2, bush, _t(az), altitude, SCALE)
        assert torch.equal(sh, sh_b[p]), f"sh differs for azimuth {az}"
        assert torch.equal(vegsh, vegsh_b[p]), f"vegsh differs for azimuth {az}"
        assert torch.equal(vbshvegsh, vbsh_b[p]), f"vbshvegsh differs for azimuth {az}"
