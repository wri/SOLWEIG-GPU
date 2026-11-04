"""Shared helpers for the A/B harness scripts.

Import this module before solweig_gpu: use_version() decides which source tree
the package is imported from. Run the scripts with any Python environment that
has the package's requirements; solweig_gpu itself need not be installed.
"""
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# The checkout this harness lives in (tools/ab_harness/_common.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Exact return order of svf_calculator (and of utci_process.load_cached_svf_outputs).
NAMES = ["svf", "svfaveg", "svfE", "svfEaveg", "svfEveg", "svfN", "svfNaveg",
         "svfNveg", "svfS", "svfSaveg", "svfSveg", "svfveg", "svfW", "svfWaveg",
         "svfWveg", "vegshmat", "vbshvegshmat", "shmat", "SVFtotal"]

# Upstream v2.0.0 fixed the north test in svf_calculator (`270 <= azimuth < 90` is
# never true); before the fix these three arrays stayed all zero.
N_ARRAYS = ("svfN", "svfNveg", "svfNaveg")


def add_version_args(ap):
    """Add --version {orig,cur} and --orig-root to an argparse parser."""
    ap.add_argument("--version", choices=["orig", "cur"], required=True,
                    help="cur: the checkout this harness is in; orig: the tree at --orig-root")
    ap.add_argument("--orig-root", default=None,
                    help="root of the reference checkout for --version orig, e.g. made with "
                         "`git worktree add ../solweig-ref origin/main`")


def use_version(version, orig_root=None):
    """Import solweig_gpu from the tree that `version` names, and check where it came from.

    cur imports from REPO_ROOT, orig from orig_root. The root goes first on
    sys.path, so no install is needed and an installed copy cannot leak in;
    worker processes (forkserver/spawn) inherit sys.path.
    Returns (solweig_gpu module, root, git revision or None).
    """
    if version == "cur":
        root = REPO_ROOT
    elif not orig_root:
        sys.exit("--version orig needs --orig-root, the root of the reference checkout "
                 "(e.g. `git worktree add ../solweig-ref origin/main`)")
    else:
        root = Path(orig_root).expanduser().resolve()
    if not (root / "solweig_gpu" / "__init__.py").is_file():
        sys.exit(f"no solweig_gpu package in {root}")
    sys.path.insert(0, str(root))
    import solweig_gpu
    actual = os.path.realpath(solweig_gpu.__file__)
    assert actual.startswith(str(root) + os.sep), \
        f"import resolved to {actual}, expected under {root}"
    return solweig_gpu, str(root), git_rev(root)


def git_rev(root):
    """`git describe --always --dirty` of root, or None outside git."""
    try:
        out = subprocess.run(["git", "-C", str(root), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def sync():
    """Wait for queued CUDA work so timings are honest (no-op without CUDA)."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def max_mem_gb():
    """Peak CUDA memory allocated by this process in GB, or None without CUDA."""
    import torch
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None


def tile_paths(data, key):
    """(Building_DSM, Trees, DEM) paths of tile `key` in a processed_inputs folder."""
    return (os.path.join(data, "Building_DSM", f"Building_DSM_{key}.tif"),
            os.path.join(data, "Trees", f"Trees_{key}.tif"),
            os.path.join(data, "DEM", f"DEM_{key}.tif"))


def require_files(*paths):
    """Exit with a clear message if any input file is missing."""
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        sys.exit("missing input file(s):\n  " + "\n  ".join(missing))


def savez_fast(path, **arrays):
    """Lossless npz (valid for np.load) with fast deflate; harness-only helper."""
    import numpy as np
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for name, arr in arrays.items():
            with zf.open(name + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(arr), allow_pickle=False)


def n_branch_dead(get):
    """True if svfN, svfNveg and svfNaveg are all zero, i.e. the tree that produced
    them predates upstream's svfN fix. `get(name)` returns the array for a name."""
    import numpy as np
    return all(not np.any(get(name)) for name in N_ARRAYS)
