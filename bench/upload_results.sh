#!/usr/bin/env bash
# Upload benchmark results JSONs to S3, under $BENCH_S3_URI/results/.
#
#   bench/upload_results.sh                   # everything in bench_runs/results/ (new or changed files only)
#   bench/upload_results.sh FILE.json [...]   # specific files
#
# Then rebuilds the HTML pages on S3 with bench/publish_report.sh (skip with --no-report).
# Set BENCH_S3_URI in bench/.env (copy bench/.env.example), or in the environment.
# Uses the AWS CLI with your usual credentials; set AWS_PROFILE there too if needed.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f bench/.env ]]; then set -a; . bench/.env; set +a; fi
dest="${BENCH_S3_URI:?set BENCH_S3_URI in bench/.env (see bench/.env.example)}"
dest="${dest%/}/results/"

report=1
if [[ "${1:-}" == "--no-report" ]]; then report=0; shift; fi

if [[ $# -eq 0 ]]; then
    aws s3 sync bench_runs/results/ "$dest" --exclude '*' --include '*.json' \
        --content-type application/json
else
    for f in "$@"; do
        aws s3 cp --only-show-errors --content-type application/json "$f" "$dest"
        echo "uploaded $f -> $dest$(basename "$f")"
    done
fi

if [[ $report -eq 1 ]]; then bench/publish_report.sh; fi
