# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "solweig-gpu",
#     "psutil",
#     "rasterio",
#     "numba",
#     "pyproj",
# ]
#
# [tool.uv.sources]
# solweig-gpu = { path = "..", editable = true }
# ///
"""
Stage-level benchmark for the SOLWEIG-GPU pipeline.

Runs the same stages as ``thermal_comfort`` one at a time on a fresh copy of the
inputs, and records wall time, CPU/memory and GPU usage for each stage together
with a description of the machine. Results are written as one JSON file per run
so runs on different platforms can be compared side by side.

GDAL's Python bindings must match the system libgdal, so pass them in at run
time (see bench/README.md):

    uv run --with "gdal==$(gdal-config --version)" bench/bench.py \\
        --input-dir /path/to/Input_rasters --era5-dir /path/to/ERA-5
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

STAGES = ["wind_coeff", "preprocess", "walls_aspect", "svf", "utci"]
INPUT_RASTERS = ["Building_DSM.tif", "DEM.tif", "Trees.tif", "Landcover.tif", "Buildings.tif"]
ZENODO = "https://zenodo.org/api/records/21081622/files"
DEFAULT_INPUT_URL = f"{ZENODO}/Input_rasters.zip/content"
DEFAULT_ERA5_URL = f"{ZENODO}/ERA-5_forcing_data.zip/content"
JSON_BEGIN = "===BENCH_JSON_BEGIN==="
JSON_END = "===BENCH_JSON_END==="


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, help="Folder with Building_DSM.tif, DEM.tif, Trees.tif[, Landcover.tif]")
    p.add_argument("--era5-dir", type=Path, help="Folder with the ERA5 NetCDF files")
    p.add_argument("--download", action="store_true",
                   help="Download and unzip the sample data as part of the run and time it "
                        "(replaces --input-dir/--era5-dir)")
    p.add_argument("--input-url", default=DEFAULT_INPUT_URL, help="Zip of input rasters for --download (default: Zenodo sample)")
    p.add_argument("--era5-url", default=DEFAULT_ERA5_URL, help="Zip of ERA5 files for --download (default: Zenodo sample)")
    p.add_argument("--date", default="2020-08-13", help="Simulation date (YYYY-MM-DD)")
    p.add_argument("--start", default="2020-08-13 00:00:00",
                   help="Start time, UTC. Together with --end it must cover --date in *local* time.")
    p.add_argument("--end", default="2020-08-14 23:00:00", help="End time, UTC")
    p.add_argument("--tile-size", type=int, default=1000)
    p.add_argument("--overlap", type=int, default=100)
    p.add_argument("--crop", type=int, metavar="N",
                   help="Benchmark only an N x N pixel window from the centre of the domain")
    p.add_argument("--stages", default=",".join(STAGES),
                   help=f"Comma-separated stages to run (default: all). Choices: {','.join(STAGES)}")
    p.add_argument("--threads", type=int, help="Set OMP/MKL/torch/numba thread counts")
    p.add_argument("--work-dir", type=Path, default=Path("bench_runs"),
                   help="Each run gets a fresh subfolder here (default: ./bench_runs)")
    p.add_argument("--label", default="", help="Free-text label stored in the results, e.g. 'm5-cpu'")
    p.add_argument("--sample-interval", type=float, default=1.0, help="Resource sampling interval in seconds")
    p.add_argument("--keep-work", action="store_true", help="Keep the run's working files (default: delete)")
    p.add_argument("--print-json", action="store_true",
                   help="Also print the results (gzipped, base64) to stdout between marker lines, so "
                        "they can be collected from a remote VM (used by coiled_bench.sh)")
    args = p.parse_args()
    if not args.download and not (args.input_dir and args.era5_dir):
        p.error("give --input-dir and --era5-dir, or --download")
    return args


# --------------------------------------------------------------------------- #
# Machine description
# --------------------------------------------------------------------------- #
def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _cpu_model() -> str:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _ec2_instance_type() -> str | None:
    """Return the EC2 instance type via IMDSv2, or None when not on EC2."""
    import urllib.request

    base = "http://169.254.169.254/latest"
    try:
        token_req = urllib.request.Request(
            f"{base}/api/token", method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(token_req, timeout=0.5).read().decode()
        req = urllib.request.Request(f"{base}/meta-data/instance-type",
                                     headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=0.5).read().decode()
    except OSError:
        return None


def machine_info() -> dict:
    import numpy as np
    import psutil
    import torch
    from osgeo import gdal

    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "ec2_instance_type": _ec2_instance_type(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "numpy": np.__version__,
        "gdal": gdal.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")},
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(gpu_name=props.name, gpu_mem_gb=round(props.total_memory / 1e9, 1),
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    # The package only selects CUDA or CPU, so this is the device it will use.
    info["solweig_device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return info


# --------------------------------------------------------------------------- #
# Resource sampling
# --------------------------------------------------------------------------- #
class Sampler(threading.Thread):
    """Samples CPU, memory and GPU usage in the background, tagged by stage."""

    def __init__(self, interval: float):
        super().__init__(daemon=True)
        import psutil

        self.interval = interval
        self.proc = psutil.Process()
        self.stage: str | None = None
        self.samples: list[dict] = []
        self._stop_event = threading.Event()
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None

    def _gpu(self) -> dict:
        if not self._has_nvidia_smi:
            return {}
        out = _run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits", "-i", "0"])
        try:
            util, mem = (float(x) for x in out.split(","))
            return {"gpu_util": util, "gpu_mem_mb": mem}
        except ValueError:
            return {}

    def _rss_tree_mb(self) -> float:
        import psutil

        rss = 0
        for p in [self.proc, *self.proc.children(recursive=True)]:
            try:
                rss += p.memory_info().rss
            except psutil.Error:
                pass
        return rss / 1e6

    def run(self) -> None:
        import psutil

        psutil.cpu_percent(percpu=True)  # prime the counters
        while not self._stop_event.wait(self.interval):
            per_cpu = psutil.cpu_percent(percpu=True)
            self.samples.append({
                "t": time.time(),
                "stage": self.stage,
                "cpu_pct": sum(per_cpu) / len(per_cpu),       # 0-100, whole machine
                "cpu_busy_cores": sum(per_cpu) / 100.0,       # e.g. 1.0 = one core fully busy
                "rss_mb": self._rss_tree_mb(),                # this process + worker processes
                **self._gpu(),
            })

    def stop(self) -> None:
        self._stop_event.set()
        self.join()

    def summarise(self, stage: str) -> dict:
        rows = [s for s in self.samples if s["stage"] == stage]
        if not rows:
            return {}

        def stat(key, fn):
            vals = [r[key] for r in rows if key in r]
            return round(fn(vals), 1) if vals else None

        mean = lambda v: sum(v) / len(v)  # noqa: E731
        return {
            "n_samples": len(rows),
            "cpu_pct_mean": stat("cpu_pct", mean),
            "cpu_busy_cores_mean": stat("cpu_busy_cores", mean),
            "cpu_busy_cores_max": stat("cpu_busy_cores", max),
            "rss_mb_max": stat("rss_mb", max),
            "gpu_util_mean": stat("gpu_util", mean),
            "gpu_util_max": stat("gpu_util", max),
            "gpu_mem_mb_max": stat("gpu_mem_mb", max),
        }


# --------------------------------------------------------------------------- #
# Data download
# --------------------------------------------------------------------------- #
def _find_dir_with(root: Path, filename: str) -> Path:
    hits = sorted(p.parent for p in root.rglob(filename) if "__MACOSX" not in p.parts)
    if not hits:
        sys.exit(f"{filename} not found in downloaded data under {root}")
    return hits[0]


def download_data(input_url: str, era5_url: str, dest: Path) -> tuple[dict, Path, Path]:
    """Download and unzip the input rasters and ERA5 files, timing each step."""
    import urllib.request
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    timings = {"files": []}
    t_all = time.perf_counter()
    for url in (input_url, era5_url):
        zip_path = dest / (url.rstrip("/").removesuffix("/content").rsplit("/", 1)[-1] or "data.zip")
        print(f"[bench] downloading {url}")
        t0 = time.perf_counter()
        with urllib.request.urlopen(url, timeout=60) as r, open(zip_path, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        dl = time.perf_counter() - t0
        size = zip_path.stat().st_size
        t0 = time.perf_counter()
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
        unzip = time.perf_counter() - t0
        zip_path.unlink()
        timings["files"].append({
            "url": url, "bytes": size, "download_seconds": round(dl, 2),
            "mb_per_s": round(size / 1e6 / dl, 1) if dl else None, "unzip_seconds": round(unzip, 2),
        })
        print(f"[bench]   {size / 1e6:.1f} MB in {dl:.1f} s ({size / 1e6 / dl:.1f} MB/s), unzip {unzip:.1f} s")
    timings["download_seconds"] = round(sum(f["download_seconds"] for f in timings["files"]), 2)
    timings["unzip_seconds"] = round(sum(f["unzip_seconds"] for f in timings["files"]), 2)
    timings["total_seconds"] = round(time.perf_counter() - t_all, 2)
    input_dir = _find_dir_with(dest, "Building_DSM.tif")
    era5_dir = _find_dir_with(dest, "data_stream-oper_stepType-instant.nc")
    return timings, input_dir, era5_dir


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def stage_inputs(src: Path, dst: Path, crop: int | None) -> dict:
    """Copy (optionally centre-cropped) input rasters into a fresh folder.

    Only the base rasters are copied, so no WindCoeff, SVF or tile outputs from
    earlier runs can be picked up and skew the timings.
    """
    from osgeo import gdal

    gdal.UseExceptions()
    dst.mkdir(parents=True)
    shape = None
    for name in INPUT_RASTERS:
        f = src / name
        if not f.exists():
            continue
        if crop:
            ds = gdal.Open(str(f))
            xoff = max(0, (ds.RasterXSize - crop) // 2)
            yoff = max(0, (ds.RasterYSize - crop) // 2)
            gdal.Translate(str(dst / name), ds, srcWin=[xoff, yoff, crop, crop])
            ds = None
        else:
            shutil.copy2(f, dst / name)
        ds = gdal.Open(str(dst / name))
        shape = (ds.RasterYSize, ds.RasterXSize)
        ds = None
    if shape is None:
        sys.exit(f"No input rasters found in {src}")
    return {"rows": shape[0], "cols": shape[1],
            "landcover": (dst / "Landcover.tif").exists()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    # Thread settings must be in the environment before numpy/torch are imported.
    if args.threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ[var] = str(args.threads)

    try:
        from osgeo import gdal  # noqa: F401
    except ImportError:
        sys.exit('GDAL Python bindings not found. Run with: uv run --with "gdal==$(gdal-config --version)" bench/bench.py ...')

    import torch
    import solweig_gpu

    if args.threads:
        torch.set_num_threads(args.threads)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        sys.exit(f"Unknown stage(s): {', '.join(sorted(unknown))}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.work_dir / run_id).resolve()
    input_dir = run_dir / "Input_rasters"
    preprocess_dir = input_dir / "processed_inputs"

    print(f"[bench] run {run_id} -> {run_dir}")
    data_timings = None
    if args.download:
        data_timings, args.input_dir, args.era5_dir = download_data(
            args.input_url, args.era5_url, run_dir / "download")
    domain = stage_inputs(args.input_dir.resolve(), input_dir, args.crop)
    print(f"[bench] domain {domain['rows']} x {domain['cols']} px")

    info = machine_info()
    print(f"[bench] {info['cpu_model']} | {info['cpu_logical']} logical CPUs | "
          f"device={info['solweig_device']} {info.get('gpu_name', '')}")

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    windcoeff_folder = None

    def run_stage(name: str) -> None:
        nonlocal windcoeff_folder
        if name == "wind_coeff":
            solweig_gpu.build_wind_ext_coeff(str(input_dir), str(args.era5_dir.resolve()))
            windcoeff_folder = str(input_dir)
        elif name == "preprocess":
            solweig_gpu.preprocess(
                base_path=str(input_dir),
                selected_date_str=args.date,
                landcover_filename="Landcover.tif" if domain["landcover"] else None,
                windcoeff_folder=windcoeff_folder,
                tile_size=args.tile_size,
                overlap=args.overlap,
                use_own_met=False,
                start_time=args.start,
                end_time=args.end,
                data_source_type="ERA5",
                data_folder=str(args.era5_dir.resolve()),
                preprocess_dir=str(preprocess_dir),
            )
        elif name == "walls_aspect":
            solweig_gpu.run_walls_aspect(str(preprocess_dir))
        elif name == "svf":
            solweig_gpu.calculate_svf(str(preprocess_dir), patch_option=2, overwrite=True)
        elif name == "utci":
            solweig_gpu.run_utci_tiles(
                base_path=str(input_dir),
                preprocess_dir=str(preprocess_dir),
                selected_date_str=args.date,
            )
            # run_utci_tiles silently skips tiles without met data, e.g. when the
            # UTC window does not cover the local date. Don't report that as a timing.
            out_root = input_dir / "output_folder"
            if not out_root.exists() or not any(p.glob("UTCI_*.tif") for p in out_root.iterdir()):
                raise RuntimeError("no UTCI tiles were produced; check that --start/--end cover "
                                   "--date in local time")

    sampler = Sampler(args.sample_interval)
    sampler.start()
    results = []
    total_t0 = time.perf_counter()
    error = None
    try:
        for name in stages:
            print(f"[bench] ---- {name} ----")
            sampler.stage = name
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            run_stage(name)
            sync()
            elapsed = time.perf_counter() - t0
            sampler.stage = None
            entry = {"stage": name, "seconds": round(elapsed, 2)}
            if torch.cuda.is_available():
                entry["torch_peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            results.append(entry)
            print(f"[bench] {name}: {elapsed:.1f} s")
    except Exception as e:  # record partial results rather than losing them
        error = f"{type(e).__name__}: {e}"
        print(f"[bench] FAILED in stage {sampler.stage}: {error}")
    finally:
        total = time.perf_counter() - total_t0
        sampler.stop()

    for entry in results:
        entry["resources"] = sampler.summarise(entry["stage"])

    n_tiles = len(list((preprocess_dir / "Building_DSM").glob("*.tif"))) if preprocess_dir.exists() else None
    n_timesteps = None
    met_files = sorted((preprocess_dir / "metfiles").glob("*.txt")) if preprocess_dir.exists() else []
    if met_files:
        n_timesteps = max(0, len(met_files[0].read_text().splitlines()) - 1)
    report = {
        "run_id": run_id,
        "label": args.label,
        "machine": info,
        "config": {
            "date": args.date, "start": args.start, "end": args.end,
            "tile_size": args.tile_size, "overlap": args.overlap, "crop": args.crop,
            "threads": args.threads, "stages": stages,
            "domain_rows": domain["rows"], "domain_cols": domain["cols"],
            "landcover": domain["landcover"], "n_tiles": n_tiles, "n_timesteps": n_timesteps,
        },
        "data_download": data_timings,  # not included in total_seconds
        "total_seconds": round(total, 2),
        "stages": results,
        "error": error,
        "samples": sampler.samples,
    }

    out_dir = args.work_dir.resolve() / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.label}" if args.label else ""
    out_file = out_dir / f"{run_id}{tag}.json"
    out_file.write_text(json.dumps(report, indent=2, default=str))

    print("\n[bench] summary")
    if data_timings:
        print(f"  data download {data_timings['download_seconds']:.1f} s, "
              f"unzip {data_timings['unzip_seconds']:.1f} s (not in total)")
    print(f"  {'stage':<14}{'seconds':>10}{'share':>8}{'cores':>8}{'rss GB':>8}{'gpu %':>7}")
    for r in results:
        res = r["resources"]
        gpu = res.get("gpu_util_mean")
        print(f"  {r['stage']:<14}{r['seconds']:>10.1f}{r['seconds'] / total:>8.0%}"
              f"{res.get('cpu_busy_cores_mean') or 0:>8.1f}{(res.get('rss_mb_max') or 0) / 1e3:>8.1f}"
              f"{'' if gpu is None else f'{gpu:.0f}':>7}")
    print(f"  {'total':<14}{total:>10.1f}")
    print(f"[bench] results -> {out_file}")
    if args.print_json:
        # gzip + base64 in short lines: `coiled run` re-wraps long output lines and
        # strips [bracketed] text as markup, which would corrupt raw JSON.
        import base64
        import gzip
        import textwrap

        payload = base64.b64encode(gzip.compress(json.dumps(report, default=str).encode())).decode()
        print(JSON_BEGIN)
        print("\n".join(textwrap.wrap(payload, 64)))
        print(JSON_END)

    if not args.keep_work:
        shutil.rmtree(run_dir, ignore_errors=True)
    if error:
        sys.exit(1)


if __name__ == "__main__":
    main()
