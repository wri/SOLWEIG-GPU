#!/usr/bin/env bash
# Run bench.py on one or more EC2 instance types with `coiled run`, and save each
# run's results JSON to bench_runs/results/ locally.
#
#   bench/coiled_bench.sh g4dn.xlarge c7i.2xlarge -- --crop 1000
#
# Instance types go before `--`; everything after it is passed to bench.py.
# Runs start in parallel, one VM each. Logs go to bench_runs/logs/.
# Needs the software environment from bench/coiled-env.yml (see bench/README.md).
# Set COILED_WORKSPACE in bench/.env (copy bench/.env.example) or the environment.
# Set SOFTWARE to override the Coiled software environment.
# If BENCH_S3_URI is set (in bench/.env), each results JSON is also uploaded to S3
# with bench/upload_results.sh, and the HTML pages are rebuilt once all runs finish.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f bench/.env ]]; then set -a; . bench/.env; set +a; fi
SOFTWARE="${SOFTWARE:-solweig-bench}"
WORKSPACE="${COILED_WORKSPACE:?set COILED_WORKSPACE in bench/.env (see bench/.env.example)}"

vm_types=()
while [[ $# -gt 0 && "$1" != "--" ]]; do vm_types+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift
bench_args=("$@")
if [[ ${#vm_types[@]} -eq 0 ]]; then
    echo "usage: $0 VM_TYPE [VM_TYPE ...] [-- bench.py args]" >&2
    exit 2
fi

mkdir -p bench_runs/logs bench_runs/results
stamp="$(date -u +%Y%m%dT%H%M%SZ)"

run_one() {
    local vm="$1" log="bench_runs/logs/${stamp}_$1.log"
    local gpu_flag=()
    [[ "$vm" == g* || "$vm" == p* ]] && gpu_flag=(--gpu)
    echo "[coiled_bench] $vm: starting (log: $log)"
    if uvx coiled run \
        --workspace "$WORKSPACE" \
        --software "$SOFTWARE" \
        --vm-type "$vm" "${gpu_flag[@]}" \
        --name "solweig-bench-${vm//./-}-$stamp" \
        --tag project=solweig-bench \
        --file bench --file solweig_gpu \
        -- env PYTHONPATH=. python bench/bench.py \
            --download --label "$vm" --print-json "${bench_args[@]}" \
        >"$log" 2>&1; then
        status=ok
    else
        status="failed (exit $?)"
    fi
    local out="bench_runs/results/${stamp}_${vm}.json"
    if python3 - "$log" "$out" <<'PY' 2>/dev/null; then
import base64, gzip, json, re, sys
text = open(sys.argv[1], errors="replace").read()
m = re.search(r"===BENCH_JSON_BEGIN===(.*?)===BENCH_JSON_END===", text, re.S)
report = json.loads(gzip.decompress(base64.b64decode(re.sub(r"[^A-Za-z0-9+/=]", "", m.group(1)))))
json.dump(report, open(sys.argv[2], "w"), indent=2)
PY
        echo "[coiled_bench] $vm: $status -> $out"
        if [[ -n "${BENCH_S3_URI:-}" ]]; then
            bench/upload_results.sh --no-report "$out" | sed "s/^/[coiled_bench] $vm: /" \
                || echo "[coiled_bench] $vm: S3 upload failed; retry with bench/upload_results.sh $out"
        fi
    else
        rm -f "$out"
        echo "[coiled_bench] $vm: $status, no results (see $log)"
    fi
}

for vm in "${vm_types[@]}"; do run_one "$vm" & done
wait

if [[ -n "${BENCH_S3_URI:-}" ]]; then
    bench/publish_report.sh || echo "[coiled_bench] report not rebuilt; retry with bench/publish_report.sh"
fi
