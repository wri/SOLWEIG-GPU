# Pipeline benchmark

`bench.py` runs the SOLWEIG-GPU pipeline one stage at a time and records, for each stage:

- wall time (with `torch.cuda.synchronize()` before stopping the timer on CUDA)
- CPU use (mean and max busy cores), peak memory of the process and its workers
- GPU utilisation and memory from `nvidia-smi`, and peak PyTorch GPU memory (CUDA only)

It also records the machine (CPU model, core counts, RAM, GPU, torch/CUDA/GDAL versions,
EC2 instance type when on EC2) and the run configuration. Each run writes one JSON file to
`bench_runs/results/`, so runs on different platforms can be compared side by side.

| Stage          | Package call                  |
|----------------|-------------------------------|
| `wind_coeff`   | `build_wind_ext_coeff`        |
| `preprocess`   | `preprocess`                  |
| `walls_aspect` | `run_walls_aspect`            |
| `svf`          | `calculate_svf`               |
| `utci`         | `run_utci_tiles`              |

Every run starts from a fresh copy of the base rasters, so cached SVF or wind-coefficient
outputs from earlier runs can't make a stage look faster than it is. The working files are
deleted afterwards unless you pass `--keep-work`.

## Running locally

GDAL's Python bindings must match the system libgdal (e.g. `brew install gdal`), so pass
them in with `--with`. The other dependencies come from the script's inline metadata.

```bash
# Download the Zenodo sample data as part of the run (download time is recorded separately)
uv run --with "gdal==$(gdal-config --version)" bench/bench.py --download --label m5-cpu

# Or use data you already have
uv run --with "gdal==$(gdal-config --version)" bench/bench.py \
    --input-dir /path/to/Input_rasters --era5-dir /path/to/ERA-5 --label m5-cpu
```

Useful options:

- `--crop N`: benchmark only an N x N pixel window from the centre of the domain, for quick iterations
- `--tile-size`, `--overlap`: tiling (default 1000 / 100)
- `--start`, `--end`: UTC window (default 2020-08-13 00:00 to 2020-08-14 23:00). The model
  runs for `--date` in *local* time, so the window must cover that local day; the
  benchmark stops with an error if no UTCI tiles are produced.
- `--stages wind_coeff,preprocess`: run only some stages (later stages need the earlier ones)
- `--threads N`: set OMP/MKL/Accelerate/numba/torch thread counts

A quick smoke test:

```bash
uv run --with "gdal==$(gdal-config --version)" bench/bench.py --download \
    --crop 400 --tile-size 400 --overlap 0 \
    --start "2020-08-13 12:00:00" --end "2020-08-13 15:00:00" --label smoke
```

## Running on EC2 with Coiled

Set your Coiled workspace in `bench/.env` (it is gitignored):

```bash
cp bench/.env.example bench/.env   # then edit COILED_WORKSPACE
```

Build the software environment once (conda-forge PyTorch with CUDA, GDAL, etc.; it also
works on CPU-only instances):

```bash
set -a; . bench/.env; set +a
uvx coiled env create --workspace "$COILED_WORKSPACE" --name solweig-bench \
    --conda bench/coiled-env.yml --gpu-enabled
```

Then run the benchmark on one or more instance types. Each gets its own VM, they run in
parallel, and each VM shuts down when its run finishes:

```bash
bench/coiled_bench.sh g4dn.xlarge g6.xlarge c7i.2xlarge -- --crop 1000
```

Runs go to the workspace set by `COILED_WORKSPACE` in `bench/.env`.
Arguments after `--` are passed to `bench.py`; `--download` and `--label <vm type>` are
added automatically. `bench/` and `solweig_gpu/` are uploaded to the VM, so the run uses
your local checkout. Each VM prints its results JSON to its log, and the script saves it to
`bench_runs/results/<stamp>_<vm type>.json` (full logs in `bench_runs/logs/`).

Suggested comparisons, each changing one thing:

| Instances                                  | Question                                        |
|--------------------------------------------|-------------------------------------------------|
| `c7i.2xlarge` vs `c7i.8xlarge`             | How do the CPU stages scale with cores?         |
| `g4dn.xlarge` vs `g6.xlarge` vs `g5.xlarge`| T4 vs L4 vs A10G, all with 4 vCPUs              |
| `g5.xlarge` vs `g5.4xlarge`                | Same GPU, more CPU: do CPU stages hold it back? |

## Looking inside a stage

The stage timings show *which* stage is slow on a platform; a profiler shows *why*.

- CPU-bound stages (`wind_coeff`, `preprocess`, `walls_aspect`, and everything on a machine
  without CUDA): sample with [py-spy](https://github.com/benfred/py-spy) and open the result
  at https://www.speedscope.app. Add `--gil` to see only time spent holding the GIL.

  Start the benchmark as usual, then attach py-spy to it from a second terminal:

  ```bash
  uv tool install py-spy
  sudo "$(which py-spy)" record -f speedscope -o prof.json --native --subprocesses \
      --pid "$(pgrep -f 'python.*bench/bench.py' | head -1)"
  ```

  Press Ctrl-C to stop recording. `sudo` is needed on macOS only; on Linux you can also
  run `py-spy record ... -- python bench/bench.py ...` directly.

- GPU stages on CUDA (`svf`, `utci`): use `torch.profiler` or `nsys profile`. CUDA runs
  asynchronously, so py-spy charges GPU time to whichever Python line waits for the
  result (often an `.item()` or `.cpu()` call).
